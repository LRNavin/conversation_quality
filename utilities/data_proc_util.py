
def clean_sample_checkpoint(samplepoint_column):
    return samplepoint_column.apply(lambda x: int(x))

def clean_timestamp(time_column):
    #TODO: Some problem with Day3, needs fixes
    return time_column.apply(
        lambda x: x.replace("PT", "").replace("H", ":").replace("M", ":").replace("S", ""))

def clean_subjects_array(subject_array):
    for i, subject in enumerate(subject_array):
        subject_array[i] = subject.split('.')[0]
    subject_array = list(filter(None, subject_array))
    return subject_array

def clean_subjects_list(subjects_column):
    split_column = subjects_column.apply(lambda x: str(x).split(" "))
    subjects_column= split_column.apply(lambda x: clean_subjects_array(x))
    return subjects_column


def clean_fformation_data(data):

    for column in data.columns:
        if column == "subjects":
            data[column] = clean_subjects_list(data[column])
        elif column == "timestart":
            data[column] = clean_timestamp(data[column])
        elif column == "timeend":
            data[column] = clean_timestamp(data[column])
        elif column == "samplestart":
            data[column] = clean_sample_checkpoint(data[column])
        elif column == "sampleend":
            data[column] = clean_sample_checkpoint(data[column])

    return data
