import json
from NNData import NNData
from LayerList import FFBPNetwork
import collections
import numpy as np

xor_data = NNData()


class MultiTypeEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, collections.deque):
            return {"__deque__": list(o)}
        elif isinstance(o, np.ndarray):
            return {"__NDarray__": o.tolist()}
        elif isinstance(o, NNData):
            return {"__NNData__": o.__dict__}
        else:
            json.JSONEncoder.default(self, o)


def multi_type_decoder(o):

    if "__deque__" in o:
        return collections.deque(o["__deque__"])
    if "__NDarray__" in o:
        return np.array(o["__NDarray__"])
    if "__NNData__" in o:
        ret_obj = NNData()
        dec_obj = o["__NNData__"]
        ret_obj._features = dec_obj["_features"]
        ret_obj._labels = dec_obj["_labels"]
        ret_obj._train_indices = dec_obj["_train_indices"]
        ret_obj._test_indices = dec_obj["_test_indices"]
        ret_obj._train_factor = dec_obj["_train_factor"]
        ret_obj._train_pool = dec_obj["_train_pool"]
        ret_obj._test_pool = dec_obj["_test_pool"]
        return ret_obj
    else:
        return o


with open("Dat.txt", "w") as f:
    json.dump(xor_data, f, cls=MultiTypeEncoder)


with open("Dat.txt", "r") as f:
    my_obj = json.load(f, object_hook=multi_type_decoder)
    print(type(my_obj))
    print(my_obj)


if __name__ == "__main__":
    XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    XOR_Y = [[0], [1], [1], [0]]
    xor_data = NNData(XOR_X, XOR_Y)
    xor_data_encoded = MultiTypeEncoder().default(xor_data)
    xor_data_decoded = multi_type_decoder(xor_data_encoded)
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(34)
    network.train(xor_data_decoded, order=NNData.Order.RANDOM)
    with open("sin_data.txt", "r") as f:
        sin_decoded = json.load(f, object_hook=multi_type_decoder)
    network.train(sin_decoded, order=NNData.Order.RANDOM)
