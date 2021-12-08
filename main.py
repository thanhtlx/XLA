import os

def table_detection(filename):
    pass


{table_detection('./forms/' + i) for i in os.listdir('./forms/') if i.endswith('.png')
 or i.endswith('.PNG') or i.endswith('.jpg') or i.endswith('.JPG')}
