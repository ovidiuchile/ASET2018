import requests

def isServerUp():
    mainAPI = 'http://localhost:8080/'
    r = requests.get(mainAPI)
    print(r.status_code)
    if r.status_code == 200:
        return True
    return False

def uploadFile():
    uploadFileAPI = 'http://localhost:8080/upload'
    files = {'image': open('star.png', 'rb')}
    r = requests.post(uploadFileAPI, files=files)
    print(r.status_code)
    if r.status_code == 200:
        return True
    return False

isServerUpTest = isServerUp()
print('test 1: {}'.format(isServerUpTest))

uploadFileTest = uploadFile()
print('test 2: {}'.format(uploadFileTest))
