from methods.polora import Polora

def get_model(model_name, args):
    name = model_name.lower()
    options = {
               'polora': Polora
               }
    return options[name](args)

