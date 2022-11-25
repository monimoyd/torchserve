import unittest

import requests
import json
import base64
from requests import Response


class TestTorchServe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://3.111.98.111:8080/predictions/cifar10/1.0"

        print(f"using base_url={cls.base_url}\n\n")

        cls.image_paths = ['airplane.png', 'automobile.png', 'bird.png', 'cat.png', 'deer.png', 'dog.png', 'frog.png', 'horse.png', 'ship.png', 'truck.png']

    def test_predict(self):
        for image_path in self.image_paths:
            print(f"testing: {image_path}")
            res = requests.post(self.base_url, files={'data': open('test_serve/' + image_path, 'rb')})
            result = res.json()
            predicted_label = list(result)[0]
            print(f"predicted label: {predicted_label}")
            actual_label = image_path.split(".")[0]

            print(f"actual label: {actual_label}")

            self.assertEqual(actual_label, predicted_label)

            print(f"done testing: {image_path}")
            print()


if __name__ == '__main__':
    unittest.main()