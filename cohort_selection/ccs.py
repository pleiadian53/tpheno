
### [todo] use a text file to define the mapping 

codeBook = {}

codeBook['CKD'] = ['585', '585.1', '585.2', '585.3', '585.4', '585.5', '585.6', '585.9', '792.5', 'V42.0', 'V45.1', 'V45.11', 
                   'V45.12', 'V56.0', 'V56.1', 'V56.2', 'V56.31', 'V56.32', 'V56.8']

# class CCS(object): 
#     @staticmethod
#     def getCohort(name='CKD'):
#         return codeBook.get(name, []) 

def getCohort(name='CKD'):
    return codeBook.get(name, []) 