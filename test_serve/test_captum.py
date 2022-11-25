import unittest
import torch
import requests
import numpy as np
import torchvision.transforms as T

from PIL import Image

from captum.attr import visualization as viz


class TestCaptum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://3.111.98.111:8080/explanations/cifar10/1.0"

        print(f"using base_url={cls.base_url}\n\n")

        cls.image_path = 'cat.png'

    def test_predict(self):
        inp_image = Image.open('test_serve/' + self.image_path)
        to_tensor = T.Compose([
	                T.Resize((224, 224)),
	                T.ToTensor()
                ])
        inp_image = to_tensor(inp_image)

        inp_image = inp_image.numpy()
        res = requests.post("http://3.111.98.111:8080/explanations/cifar10", files={'data': open('test_serve/' + self.image_path, 'rb')})
        ig=res.json()

        attributions = np.array(ig)

        inp_image, attributions = inp_image.transpose(1, 2, 0), attributions.transpose(1, 2, 0)
        self.assertEqual(inp_image.shape, attributions.shape) 

        viz.visualize_image_attr(attributions, inp_image, method="blended_heat_map",sign="all", show_colorbar=True, title="Overlayed Integrated Gradients")
    
        print(f"done testing: ")
        print()


if __name__ == '__main__':
    unittest.main()

