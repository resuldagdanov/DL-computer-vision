import os
import zipfile

_A2_FILES = [
  'linear_classifier.py',
  'linear_classifier.ipynb',
  'two_layer_net.py',
  'two_layer_net.ipynb',
  'svm_best_model.pt',
  'softmax_best_model.pt',
  'nn_best_model.pt',
]


def make_a2_submission(assignment_path: str,
                       uniquename: str=None,
                       umid: str=None):
  _make_submission(assignment_path, _A2_FILES, 'A2', uniquename, umid)


def _make_submission(assignment_path: str,
                     file_list: list,
                     assignment_no: str,
                     uniquename: str=None,
                     umid: str=None):
  
  if uniquename is None or umid is None:
    uniquename, umid = _get_user_info(uniquename, umid)
  
  zip_path = '{}_{}_{}.zip'.format(uniquename, umid, assignment_no)
  zip_path = os.path.join(assignment_path, zip_path)
  
  print('Writing zip file to: ', zip_path)
  
  with zipfile.ZipFile(zip_path, 'w') as zf:
  
    for filename in file_list:
      in_path = os.path.join(assignment_path, filename)
  
      if not os.path.isfile(in_path):
        raise ValueError('Could not find file "%s"' % filename)

      zf.write(in_path, filename)


def _get_user_info(uniquename: str=None,
                   umid: str=None):
  if uniquename is None:
    uniquename = input('Enter your uniquename (e.g. justincj): ')

  if umid is None:
    umid = input('Enter your umid (e.g. 12345678): ')

  return uniquename, umid
