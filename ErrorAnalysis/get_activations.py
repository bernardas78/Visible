from tensorflow.keras.backend import function

def get_activations(model, X):
    #print("Getting activations of X shaped ", X.shape)

    # Initialize a return value: list of activations
    list_activations = []
    list_layernames = []

    # Don't include input layers because full original picture will be included
    ## include input layer as well
    # list_activations.append ( X )
    # list_layernames.append ( 'Input' )

    for layer in model.layers:

        # only show cnn and maxpooling layers
        #if not 'conv' in layer.name and not 'pool' in layer.name:
        #    continue
        #print("Getting activations of ", layer.name)

        # Get a function to calculate output of the layer
        func_activation = function([model.input], [layer.output])

        # Calculate activation layer's output
        output_activation = func_activation([X])[0]

        # Append output to a list of activations for return
        list_activations.append(output_activation)
        list_layernames.append(layer.name)

    return (list_layernames, list_activations)

# Pre-last layer activations
def get_activations_preLast(model, X):

    layer = model.layers[-2]
    func_activation = function([model.input], [layer.output])
    output_activation = func_activation([X])[0]

    return output_activation

# Last layer activations
def get_activations_Last(model, X):

    layer = model.layers[-1]
    func_activation = function([model.input], [layer.output])
    output_activation = func_activation([X])[0]

    return output_activation