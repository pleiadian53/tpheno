


### 
#  CKD labels in the dataset: eMerge_NKF_Stage_20170818.csv
# 
#  ['CKD Stage 3a', 'Unknown', 'CKD Stage 3b', 'ESRD on dialysis', 'CKD G1-control', 'CKD G1A1-control', 
#   'CKD Stage 5', 'CKD Stage 4', 'ESRD after transplant', 'CKD Stage 2', 'CKD Stage 1']
# 
# 

###
#
# Chronic Kidney Disease according to Clinical Classification Software (CCS)
#     585 585.1 585.2 585.3 585.4 585.5 585.6 585.9 792.5 V42.0 V45.1 V45.11 V45.12 V56.0 V56.1 V56.2 V56.31 V56.32 V56.8 
#
#

GDiagnoses = {'585', '585.1', '585.2', 
              '585.3', '585.4', '585.5',
              '585.6', '585.9', '792.5', 
              }  # 'V42.0', 'V45.1', 'V45.11', 'V45.12', 'V56.0', 'V56.1', 'V56.2', 'V56.31', 'V56.32', 'V56.8'

GDiagnosesFull = {'585', '585.1', '585.2', 
                  '585.3', '585.4', '585.5',
                  '585.6', '585.9', '792.5', 
                  'V42.0', 'V45.1', 'V45.11', 'V45.12', 'V56.0', 'V56.1', 'V56.2', 'V56.31', 'V56.32', 'V56.8', 

                  # '585.00', '585.10', '585.30', '', # error correction
              } 

GDiagnosesApprox = {
    '585.0', '585.00', '585.10', '585.20', '585.30', '585.40', '585.50', '585.60', '585.90', '792.50', 
    'V42.00', 'V45.10', 'V56.00', 'V56.10', 'V56.20', 'V56.80', 
}

GDiagnosesApprox = GDiagnosesApprox.union(GDiagnosesFull)


def isCase(code): 
    """
    Use a set of ICD-9 or ICD-10 to identify a target disease. 

    Memo
    ----
    1. 792.5: Cloudy (hemodialysis) (peritoneal) dialysis effluent
    """
    scode = str(code).strip()
    if scode in GDiagnoses: 
        return True
    # loose definition 
    if scode.startswith(('585', )): 
        return True 
    return False

def isCaseCCS(code):
    scode = str(code).strip()
    # if scode in GDiagnosesApprox: 
    if scode in GDiagnosesFull: 
        return True
    if scode.startswith(('585', )): 
        return True 
    return False