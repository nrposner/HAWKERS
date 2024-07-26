import pandas as pd

category_map = {
    'costumerservice': "Customer Service",
    'delivery-service': "Delivery Service",
    'delivery-service,costumerservice': "Mixed",
    'delivery-service,product-quality': "Mixed",
    'delivery-service,web-review,product-quality': "Mixed",
    'devolucion': "Customer Service",
    'devolucion,product-quality': "Mixed",
    'product-quality': "Product Quality",
    'product-quality,delivery-service': "Mixed",
    'product-quality,web-review': "Product Quality",
    'producto': "Product Quality",
    'serviciopost-venta': "Customer Service",
    'servicioposventa': "Customer Service",
    'servicioposventa,delivery-service': "Mixed",
    'web-review': "Other",
    'costumerservice,': "Customer Service",
    'costumerservice,delivery-service': "Mixed",
    'costumerservice,delivery-service,': "Mixed",
    'costumerservice,delivery-service,product-quality': "Mixed",
    'costumerservice,devolucion': "Customer Service",
    'costumerservice,devolucion,': "Customer Service",
    'costumerservice,product-quality': "Mixed",
    'costumerservice,product-quality,delivery-service': "Mixed",
    'costumerservice,product-quality,serviciopost-venta': "Mixed",
    'costumerservice,producto': "Mixed",
    'costumerservice,producto,': "Mixed",
    'costumerservice,producto,serviciopost-venta': "Mixed",
    'costumerservice,producto,serviciopost-venta,': "Mixed",
    'costumerservice,serviciopost-venta': "Customer Service",
    'delivery-service,costumerservice,product-quality': "Mixed",
    'delivery-service,devolucion': "Delivery Service",
    'delivery-service,p,product-quality': "Mixed",
    'delivery-service,product-quality,costumerservice': "Mixed",
    'delivery-service,product-quality,devolucion': "Mixed",
    'delivery-service,product-quality,serviciopost-venta': "Mixed",
    'delivery-service,product-quality,servicioposventa': "Mixed",
    'delivery-service,product-quality,web-review': "Mixed",
    'delivery-service,producto': "Mixed",
    'delivery-service,serviciopost-venta': "Mixed",
    'delivery-service,serviciopost-venta,product-quality': "Mixed",
    'delivery-service,servicioposventa': "Mixed",
    'delivery-service,web-review': "Delivery Service",
    'devolucion,': "Customer Service",
    'devolucion,costumerservice': "Customer Service",
    'devolucion,producto': "Mixed",
    'devolucion,serviciopost-venta': "Customer Service",
    'manual-send': "Other",
    'manual-send,servicioposventa': "Customer Service",
    'product-quality,': "Product Quality",
    'product-quality,costumerservice': "Mixed",
    'product-quality,delivery-service,web-review':"Mixed",
    'product-quality,devolucion': "Mixed",
    'product-quality,serviciopost-venta': "Mixed",
    'product-quality,servicioposventa': "Mixed",
    'product-quality,web-review,delivery-service': "Mixed",
    'product-quality,web-review,servicioposventa': "Mixed",
    'producto,costumerservice': "Mixed",
    'producto,delivery-service': "Mixed",
    'producto,product-quality': "Product Quality",
    'producto,serviciopost-venta': "Mixed",
    'serviciopost-venta,': "Customer Service",
    'serviciopost-venta,delivery-service': "Mixed",
    'serviciopost-venta,product-quality': "Mixed",
    'serviciopost-venta,spam': "Other",
    'servicioposventa,product-quality': "Mixed",
    'spam': "Other",
    'web-review,delivery-service': "Delivery Service",
    'web-review,delivery-service,product-quality': "Mixed",
    'web-review,product-quality': "Product Quality",
    'web-review,product-quality,delivery-service': "Mixed",
    'web-review,producto': "Product Quality",
    'web-review,serviciopost-venta': "Customer Service"
    
}


def tags_conversion(data_tags, category_map):
    """In order to effectively classify, we must convert the native Trustpilot tags into 
    our desired categories: 'Product Quality', 'Customer Service', 'Delivery Service', 'Mixed', and 'Other'
    The function below should run on the tags column of the dataframe exclusively

    We accomplish this with a dictionary of existing category combinations and the desired replacement
    
    This is rather ad-hoc, since the exisitng tags do not map neatly onto our desired categories. As data expands, 
    it will be necessary to add new entries to the category map above, and perhaps to change the existing entries
    in light of a more comprehensive conversion schema. 
    
    """

    data_tags.replace(category_map, inplace=True)

    data_tags.reset_index(inplace=True, drop=True)

    return data_tags
