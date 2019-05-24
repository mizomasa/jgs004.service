import base64

def encode(file_path):
    #file読み込み
    with open(file_path, "rb") as file:
        return encode_data(file.read())

def encode_data(data):
    return base64.b64encode(data)

def decode(base64_data):
    return base64.b64decode(base64_data)

def decode_with_save(base64_data, outfile_path):
    with open(outfile_path, "wb") as file:
        file.write(decode(base64_data))
