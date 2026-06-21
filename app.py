import os
import json
import torch
from net import build_model, predict_probs, topk_probs
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LABELS_PATH = os.path.join(MODELS_DIR, 'labels.json')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pth')
STATIC_GALLERY_DIR = os.path.join(BASE_DIR, 'static', 'gallery')

# 中文映射映射
EN_TO_ZH = {
    'rose': '玫瑰', 'chinese_rose': '月季', 'lily': '百合', 'carnation': '康乃馨',
    'sunflower': '向日葵', 'chrysanthemum': '菊花', 'lotus': '荷花',
    'tree_peony': '牡丹', 'herbaceous_peony': '芍药', 'plum_blossom': '梅花', 'peach_blossom': '桃花',
    'cherry_blossom': '樱花', 'crabapple': '海棠', 'magnolia': '玉兰', 'azalea': '杜鹃',
    'bougainvillea': '三角梅', 'hydrangea': '绣球花', 'lavender': '薰衣草', 'baby_breath': '满天星',
    'forget_me_not': '勿忘我', 'hyacinth': '风信子', 'tulip': '郁金香', 'daffodil': '水仙花',
    'clivia': '君子兰', 'phalaenopsis': '蝴蝶兰', 'golden_pothos': '绿萝',
    'kalanchoe': '长寿花', 'cyclamen': '仙客来', 'dandelion': '蒲公英',
    'spider_plant': '吊兰'
}

LOW_CONFIDENCE_THRESHOLD = 65.0
MIN_MARGIN_THRESHOLD = 12.0
TOPK_RETURN = 3
EXCLUDED_CLASSES = {'water_lily'}

# 伪造一些花卉的百科数据以还原设计图展现效果
FLOWER_DATABASE = {
    '玫瑰': {'name': '玫瑰', 'family': '蔷薇科', 'alias': '刺玫花', 'flower_language': '热恋、爱情', 'location': '原产：亚洲温带，现全球栽培', 'img': '/static/gallery/rose.jpg'},
    '月季': {'name': '月季', 'family': '蔷薇科', 'alias': '月月红', 'flower_language': '等待有希望的希望', 'location': '原产：中国/喜马拉雅地区，后广泛栽培', 'img': '/static/gallery/chinese_rose.jpg'},
    '百合': {'name': '百合', 'family': '百合科', 'alias': '强瞿', 'flower_language': '纯洁、高雅', 'location': '原产：北半球温带（欧洲、亚洲、美洲）', 'img': '/static/gallery/lily.jpg'},
    '康乃馨': {'name': '康乃馨', 'family': '石竹科', 'alias': '香石竹', 'flower_language': '母爱、健康', 'location': '原产：欧洲及西亚，现世界各地栽培', 'img': '/static/gallery/carnation.jpg'},
    '向日葵': {'name': '向日葵', 'family': '菊科', 'alias': '朝阳花', 'flower_language': '信念、光辉', 'location': '原产：北美，现全球栽培', 'img': '/static/gallery/sunflower.jpg'},
    '菊花': {'name': '菊花', 'family': '菊科', 'alias': '寿客', 'flower_language': '清净、高洁', 'location': '原产：中国，现东亚广泛栽培', 'img': '/static/gallery/chrysanthemum.jpg'},
    '荷花': {'name': '荷花', 'family': '莲科', 'alias': '水芙蓉', 'flower_language': '清白、坚贞纯洁', 'location': '原产：亚洲，尤其是东亚与南亚', 'img': '/static/gallery/lotus.jpg'},
    '牡丹': {'name': '牡丹', 'family': '芍药科', 'alias': '木芍药', 'flower_language': '富贵、圆满', 'location': '原产：中国，后传入日本、朝鲜及欧洲', 'img': '/static/gallery/tree_peony.jpg'},
    '芍药': {'name': '芍药', 'family': '芍药科', 'alias': '将离', 'flower_language': '情有独钟', 'location': '原产：温带亚洲', 'img': '/static/gallery/herbaceous_peony.jpg'},
    '梅花': {'name': '梅花', 'family': '蔷薇科', 'alias': '春梅', 'flower_language': '坚强、高雅', 'location': '原产：东亚，尤以中国为代表', 'img': '/static/gallery/plum_blossom.jpg'},
    '桃花': {'name': '桃花', 'family': '蔷薇科', 'alias': '阳春花', 'flower_language': '爱情的俘虏', 'location': '原产：中国及喜马拉雅地区', 'img': '/static/gallery/peach_blossom.jpg'},
    '樱花': {'name': '樱花', 'family': '蔷薇科', 'alias': '山樱', 'flower_language': '生命、希望', 'location': '主要分布：东亚（日本、中国、朝鲜），亦有欧洲栽培品种', 'img': '/static/gallery/cherry_blossom.jpg'},
    '海棠': {'name': '海棠', 'family': '蔷薇科', 'alias': '解语花', 'flower_language': '温和、美丽', 'location': '原产：东亚及喜马拉雅地区', 'img': '/static/gallery/crabapple.jpg'},
    '玉兰': {'name': '玉兰', 'family': '木兰科', 'alias': '白玉兰', 'flower_language': '报恩、纯洁', 'location': '原产：东亚（中国、越南）', 'img': '/static/gallery/magnolia.jpg'},
    '杜鹃': {'name': '杜鹃', 'family': '杜鹃花科', 'alias': '映山红', 'flower_language': '永远属于你', 'location': '分布：北半球多山区（亚洲、欧洲、美洲）', 'img': '/static/gallery/azalea.jpg'},
    '三角梅': {'name': '三角梅', 'family': '紫茉莉科', 'alias': '叶子花', 'flower_language': '热情、坚韧', 'location': '原产：南美洲热带，现热带及亚热带广泛栽培', 'img': '/static/gallery/bougainvillea.jpg'},
    '绣球花': {'name': '绣球花', 'family': '虎耳草科', 'alias': '八仙花', 'flower_language': '希望、忠贞', 'location': '原产：亚洲与美洲部分地区，广泛庭院栽培', 'img': '/static/gallery/hydrangea.jpg'},
    '薰衣草': {'name': '薰衣草', 'family': '唇形科', 'alias': '香水植物', 'flower_language': '等待爱情', 'location': '原产：地中海地区，现全球温暖干燥地带种植', 'img': '/static/gallery/lavender.jpg'},
    '满天星': {'name': '满天星', 'family': '石竹科', 'alias': '圆锥石头花', 'flower_language': '清纯、关心', 'location': '原产：地中海及欧亚部分地区，常用于花束填充', 'img': '/static/gallery/baby_breath.jpg'},
    '勿忘我': {'name': '勿忘我', 'family': '紫草科', 'alias': '星辰花', 'flower_language': '永恒的爱', 'location': '原产：欧洲及亚洲温带地区', 'img': '/static/gallery/forget_me_not.jpg'},
    '风信子': {'name': '风信子', 'family': '风信子科', 'alias': '洋水仙', 'flower_language': '燃生命之火', 'location': '原产：东地中海至中东地区，现欧洲广泛栽培', 'img': '/static/gallery/hyacinth.jpg'},
    '郁金香': {'name': '郁金香', 'family': '百合科', 'alias': '洋荷花', 'flower_language': '博爱、体贴', 'location': '原产：中亚，16世纪传入欧洲后广泛栽培', 'img': '/static/gallery/tulip.jpg'},
    '水仙花': {'name': '水仙花', 'family': '石蒜科', 'alias': '凌波仙子', 'flower_language': '期盼爱情', 'location': '原产：地中海沿岸及西欧部分地区，后传入亚洲', 'img': '/static/gallery/daffodil.jpg'},
    '君子兰': {'name': '君子兰', 'family': '石蒜科', 'alias': '剑叶石蒜', 'flower_language': '高贵、丰盛', 'location': '原产：南非，常作为观叶观花盆栽', 'img': '/static/gallery/clivia.jpg'},
    '蝴蝶兰': {'name': '蝴蝶兰', 'family': '兰科', 'alias': '蝶兰', 'flower_language': '我爱你、幸福向你飞来', 'location': '原产：热带亚洲与澳大利亚部分地区，广泛盆栽', 'img': '/static/gallery/phalaenopsis.jpg'},
    '绿萝': {'name': '绿萝', 'family': '天南星科', 'alias': '黄金葛', 'flower_language': '守望幸福', 'location': '原产：热带亚洲，常见室内观叶植物', 'img': '/static/gallery/golden_pothos.jpg'},
    '长寿花': {'name': '长寿花', 'family': '景天科', 'alias': '圣诞伽蓝菜', 'flower_language': '大吉大利、长命百岁', 'location': '原产：非洲马达加斯加，现为常见室内盆栽', 'img': '/static/gallery/kalanchoe.jpg'},
    '仙客来': {'name': '仙客来', 'family': '报春花科', 'alias': '萝卜海棠', 'flower_language': '内向、优美', 'location': '原产：地中海地区，现作盆栽广泛栽培', 'img': '/static/gallery/cyclamen.jpg'},
    '蒲公英': {'name': '蒲公英', 'family': '菊科', 'alias': '黄花地丁', 'flower_language': '无法停留的爱', 'location': '广泛分布于北半球温带和暖温带', 'img': '/static/gallery/dandelion.jpg'},
    '吊兰': {'name': '吊兰', 'family': '天门冬科', 'alias': '折鹤兰', 'flower_language': '无奈而又给人希望', 'location': '原产：南非，现作为常见室内观叶植物', 'img': 'https://images.unsplash.com/photo-1592150621744-aca64f48394a?auto=format&fit=crop&w=800&q=80'},
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = None
class_names = []
_model_mtime = 0


def init_model(force=False):
    global model, class_names, _model_mtime
    if not os.path.exists(LABELS_PATH) or not os.path.exists(MODEL_PATH):
        return False
    mtime = os.path.getmtime(MODEL_PATH)
    if not force and model is not None and mtime == _model_mtime:
        return True

    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        class_names = json.load(f)

    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    for arch in ('resnet34', 'resnet18'):
        try:
            net = build_model(len(class_names), arch=arch)
            net.load_state_dict(state)
            from net import save_arch
            save_arch(arch)
            break
        except RuntimeError:
            continue
    else:
        return False
    model = net.to(device).eval()
    _model_mtime = mtime
    return True


init_model()

def prepare_gallery_images():
    """图鉴由 update_gallery.py 离线生成；启动时仅确保目录存在。"""
    os.makedirs(STATIC_GALLERY_DIR, exist_ok=True)

prepare_gallery_images()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gallery')
def gallery_page():
    return render_template('gallery.html')

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/api/gallery', methods=['GET'])
def get_gallery():
    zh_to_en = {zh: en for en, zh in EN_TO_ZH.items()}
    flowers = []
    for item in FLOWER_DATABASE.values():
        entry = dict(item)
        en = zh_to_en.get(item['name'])
        if en:
            path = os.path.join(STATIC_GALLERY_DIR, f'{en}.jpg')
            if os.path.isfile(path):
                v = int(os.path.getmtime(path))
                entry['img'] = f'/static/gallery/{en}.jpg?v={v}'
        flowers.append(entry)
    return jsonify(flowers)

@app.route('/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': '没有找到文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    try:
        if not init_model():
            return jsonify({'error': '模型未就绪，请先运行 train.py'}), 503

        image = Image.open(file)
        probabilities = predict_probs(model, image, device, tta=True)
        candidates_en = topk_probs(probabilities, class_names, k=len(class_names))
        candidates_en = [c for c in candidates_en if c['name'] not in EXCLUDED_CLASSES]
        if not candidates_en:
            return jsonify({'error': '可用类别为空，请重新训练模型后再试'}), 503

        candidates_en = candidates_en[:TOPK_RETURN]
        best = candidates_en[0]
        second = candidates_en[1] if len(candidates_en) > 1 else None

        confidence = best['prob'] * 100
        margin = (best['prob'] - second['prob']) * 100 if second else 100.0
        uncertain = confidence < LOW_CONFIDENCE_THRESHOLD or margin < MIN_MARGIN_THRESHOLD

        en_name = best['name']
        zh_name = EN_TO_ZH.get(en_name, en_name)
        info = FLOWER_DATABASE.get(zh_name, {
            'family': '未知', 'alias': '暂无', 'flower_language': '无', 'location': '未知',
        })

        candidates = []
        for item in candidates_en:
            c_en = item['name']
            c_zh = EN_TO_ZH.get(c_en, c_en)
            candidates.append({
                'name': c_zh,
                'confidence': f"{item['prob'] * 100:.1f}%",
                'confidence_val': item['prob'] * 100,
            })

        message = ''
        if uncertain:
            message = '结果不够确定，建议更换清晰正面图片或在自然光下重拍。'

        return jsonify({
            'success': True,
            'name': zh_name,
            'confidence': f'{confidence:.1f}%',
            'confidence_val': confidence,
            'margin': margin,
            'uncertain': uncertain,
            'message': message,
            'candidates': candidates,
            'info': info,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv('FLASK_RUN_PORT', '5000')))
