from sklearn.preprocessing import LabelEncoder

def labelToDigits(vect):
    label_encoder = LabelEncoder()
    label_encoder.fit(vect)
    vect_encoded = label_encoder.transform(vect)
    return vect_encoded