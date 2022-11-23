# test load_mnist

import unittest
import torchvision
import torchvision.datasets as datasets
from src.utils.load_mnist import load_mnist

class TestLoadMnist(unittest.TestCase):
    def test_load_mnist(self):
        batch_size = 64
        trainloader, testloader = load_mnist(batch_size)
        self.assertEqual(len(trainloader), 938)
        self.assertEqual(len(testloader), 157)
        self.assertEqual(trainloader.batch_size, batch_size)
        self.assertEqual(testloader.batch_size, batch_size)
        self.assertEqual(trainloader.dataset.train, True)
        self.assertEqual(testloader.dataset.train, False)
        self.assertEqual(trainloader.dataset.transform.transforms[0].size, (32, 32))
        self.assertEqual(testloader.dataset.transform.transforms[0].size, (32, 32))
        self.assertEqual(trainloader.dataset.transform.transforms[1].__class__.__name__, 'ToTensor')
        self.assertEqual(testloader.dataset.transform.transforms[1].__class__.__name__, 'ToTensor')
        self.assertEqual(trainloader.dataset.transform.transforms[0].__class__.__name__, 'Resize')
        self.assertEqual(testloader.dataset.transform.transforms[0].__class__.__name__, 'Resize')
        self.assertEqual(trainloader.dataset.__class__.__name__,    'MNIST')