import pandas as pd
import utilities.data_read_util as reader
import constants

# df = pd.DataFrame([10, 20, 15, 30, 45])
#
# print(df)
# print(df.shift(-2))
#
# for i in range(0,4):
#     print(i)

# reader.get_annotated_fformations(constants.fform_annot_data)

print(reader.get_accel_data_from_f_form())