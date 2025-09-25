from sklearn.neighbors import KNeighborsClassifier

X=[[100,7],[200,7.5],[250,8],[300,8.5],[330,8.5],[360,9]]
# 1 is apple 0 is orange
y=[1,0,0,0,1,1]
model=KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)
weight=float(input ("enter the weight in grams:"))
size=float(input("enter the size in cm:"))
prediction=model.predict([[weight,size]])[0]
if prediction==1:
    print("it is an apple")
else:
    print("it is an orange")