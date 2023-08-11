import numpy as np
from sklearn.model_selection import train_test_split

data_path = ''
data = np.load(data_path)
save_train_dir = ''
save_test_dir = ''

train,test=train_test_split(data,test_size=0.2,random_state=42)

def save_data(flag,name,data):
    if flag == 1:
        np.save(save_train_dir+name,data)
        print("saved "+name+"done.")
    elif flag == 0:
        np.save(save_test_dir+name,data)
        print("saved "+name+"done.")


save_data(1,'left_train.npy',train)
save_data(0,'left_test.npy',test)