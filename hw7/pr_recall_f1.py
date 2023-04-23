import numpy
def precision(y_true, y_pred):
  TP = 0
  FP = 0
  for i in range(len(y_true)):
    predict = y_pred[i]
    gr_truth= y_true[i]
    if predict == 1 and gr_truth ==1:
      TP +=1
    if predict == 1 and gr_truth == 0:
      FP +=1
  return (TP)/(TP+FP)

def recall(y_true, y_pred):
  TP = 0
  FN = 0
  for i in range(len(y_true)):
    predict = y_pred[i]
    gr_truth= y_true[i]
    if predict == 1 and gr_truth ==1:
      TP +=1
    if predict == 0 and gr_truth == 1:
      FP +=1
  return (TP)/(TP+FN)

def f1(y_true, y_pred):
  """
    TODO: Заполните тело функции вычисления f1-меры предсказания
  """
  PR = self.precision(y_true, y_pred)
  RC = self.recall(y_true, y_pred)
  return 2*(PR*RC)/(PR+RC)
