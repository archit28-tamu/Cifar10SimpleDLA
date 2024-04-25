# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

class config_class(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

model_configs = config_class({
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 2,
    "num_classes": 10,
    "learning_rate": 0.01,
    "num_classes": 10,
    "save_interval": 10
})

training_configs = config_class({
	"learning_rate": 0.01
})

### END CODE HERE