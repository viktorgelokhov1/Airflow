import json
import dill
import pandas as pd


def predict():
    with open("data/models/cars_pipe_202505131914.pkl", "rb") as f:
        model = dill.load(f)

    with open("data/test/7310993818.json", "rb") as f1:
        test1 = json.load(f1)

    df1 = pd.DataFrame.from_dict([test1])
    y1 = model.predict(df1)
    df1["price_category"] = y1[0]



    with open("data/test/7313922964.json", "rb") as f2:
        test2 = json.load(f2)

    df2 = pd.DataFrame.from_dict([test2])
    y2 = model.predict(df2)
    df2["price_category"] = y2[0]


    with open("data/test/7315173150.json", "rb") as f3:
        test3 = json.load(f3)

    df3 = pd.DataFrame.from_dict([test3])
    y3 = model.predict(df3)
    df3["price_category"] = y3[0]


    with open("data/test/7316152972.json", "rb") as f4:
        test4 = json.load(f4)

    df4 = pd.DataFrame.from_dict([test4])
    y4 = model.predict(df4)
    df4["price_category"] = y4[0]


    with open("data/test/7316509996.json", "rb") as f5:
        test5 = json.load(f5)

    df5 = pd.DataFrame.from_dict([test5])
    y5 = model.predict(df5)
    df5["price_category"] = y5[0]


    final_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    final_df.to_csv("data/predictions/Predictions.csv", index=False)








if __name__ == '__main__':
    predict()
