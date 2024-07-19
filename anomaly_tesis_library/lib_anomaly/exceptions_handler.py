## errors handling

class NoDataset(Exception):
    def __init__(self):
        # Call the base class constructor with the custom message
        super().__init__("the [dataset_name] is empty.")
 

class EmptyParams(Exception):
    def __init__(self):
        # Call the base class constructor with the custom message
        super().__init__("The **params or **kwargs is empty.")