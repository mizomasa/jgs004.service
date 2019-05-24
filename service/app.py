from flask import Flask, request, jsonify
import domain.user_face_image as fi
import repos.annoy_wrapper as aw

app = Flask(__name__, static_folder="resorce")
aw.AnnoyWrapper.get_instance()

@app.route('/test_json_get', methods=['GET'])
def test_json_get():
    print(request.args.get('hoge'))
    print(request.args.get('foo'))
    return jsonify(request.args)

@app.route('/extractFace', methods=['post'])
def extractFace():
    request_param = request.get_json()

    if request_param is None:
        return responce_as_json('error', 'invalid param. param type is json only.')
    user_image = fi.UserFaceImage()
    user_image.data_orginal_as_base64 = request_param['data_orginal_as_base64']
    user_image.extract_face().apply_to_classifier().extract_feature()

    ui_paths = aw.AnnoyWrapper.get_instance().get_ui_list(user_image.genre, user_image.feature)
    ui_paths = [v.replace('./', '/') for v in ui_paths]
    return responce_as_json('OK', 'sucess', user_image.genre, ui_paths)

def responce_as_json(status, message, genre = None, images = None):
    return jsonify({
        'status':status,
        'message':message,
        'genre':genre,
        'images':images
        })


if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    app.run(host='0.0.0.0', port=9000) # どこからでもアクセス可能に
