import unittest

import requests
import json
import base64
from requests import Response
import sys
import pyrootutils
root = pyrootutils.setup_root(sys.path[0], pythonpath=True, cwd=True)
sys.path.append('./ts_scripts')
from ts_scripts.torchserve_grpc_client import infer, get_inference_stub

class TestGrpcTorchServe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_stub = get_inference_stub()

        #print(f"using base_url={cls.base_url}\n\n")

        cls.image_paths = ['airplane.png', 'automobile.png', 'bird.png', 'cat.png', 'deer.png', 'dog.png', 'frog.png', 'horse.png', 'ship.png', 'truck.png']

    def test_predict(self):
        for image_path in self.image_paths:
            print(f"testing: {image_path}")
            res = infer(self.client_stub, 'cifar10', 'test_serve/' + image_path)
            result = json.loads(res)

            predicted_label = list(result)[0]
            print(f"predicted label: {predicted_label}")
            actual_label = image_path.split(".")[0]

            print(f"actual label: {actual_label}")

            self.assertEqual(actual_label, predicted_label)

            print(f"done testing: {image_path}")
            print()

if __name__ == '__main__':
    unittest.main()


