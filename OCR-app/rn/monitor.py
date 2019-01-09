import logging
import os


KNOWN_IMAGE_HEADERS = [b'JFIF', b'PNG', b'BMP']
KNOWN_IMAGE_EXTENSIONS = ['bmp', 'jpg', 'jpeg', 'png']
logger = logging.getLogger('user_uploads')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('user_uploads.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def exception_catcher(func):
    def inner(a, b):
        try:
            res = func(a, b)
            logger.info("Success case: %s %s", a, b)
            return res
        except BaseException as exc:
            logger.exception("Fail case: %s %s\n%s", a, b, exc)
            raise exc
    return inner


@exception_catcher
def check_valid_image(fp, original_filename):
    if os.path.getsize(fp) >= 1024**2:
        raise Exception('Image size must be lower than 1 mb')

    if not any([original_filename.lower().endswith(i) for i in KNOWN_IMAGE_EXTENSIONS]):
        raise Exception('Image extension not recognized')

    with open(fp, 'rb') as fhr:
        header = fhr.read(20)
        if not any([i in header for i in KNOWN_IMAGE_HEADERS]):
            raise Exception('Image header not recognized')
