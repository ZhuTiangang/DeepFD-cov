import pandas as pd
import numpy as np
import re

if __name__ == '__main__':
    df = pd.read_csv('../rank1.csv', header=None)
    print(df)
    print(df.shape)
    for i in range(df.shape[0]):
        # cur = df.iloc[i]
        str = ""
        # if np.isnan(df.iloc[i][0]):
        #     # print(i)
        #     continue
        # print(df.iloc[i][0])
        for j in df.iloc[i][:]:
            # print(type(j))
            if type(j) == float:
                break
            cur = j[::-1]
            # print(cur)
            cur = re.sub(r'_', '.', cur, 1)
            cur = cur[::-1]
            # print(cur[3:])
            str += cur[3:] + " + "
        # print(df.iloc[i][0])
        print(str)

