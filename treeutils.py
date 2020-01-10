def create_node(data, left = None, right = None):
    return (data, left, right)

def is_leaf(node):
    return node[1] is None and node[2] is None

def left(node):
    return node[1]

def right(node):
    return node[2]

def get_datum(node):
    return node[0]
