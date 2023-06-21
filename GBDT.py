import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np


df = pd.read_csv('ceshi/XXX.csv')
df.head()
df= df.iloc[:, 1:]
for num in range(1, 12):
    df['B' + str(num)].value_counts()
 #   df['X'].value_counts()
 #   df['Y'].value_counts()
    df['Z'].value_counts()
#print(df)


X = df.drop(columns='Z')
y = df['Z']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=123)
model.fit(X_train, y_train)

save_filename = 'ceshi/XXX.pth'
torch.save(model, save_filename)
print('Saved as %s' % save_filename)


y_pred = model.predict(X_test)



a = pd.DataFrame()  
a['P'] = list(y_pred)
a['A'] = list(y_test)
a.head()

print("score:",model.score(X_test, y_test))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, model.predict(X_test))
print("R2:",r2)

print("feature_importances：",model.feature_importances_)

features = X.columns  
importances = model.feature_importances_  
filename='ceshi/XXX.txt'
np.savetxt(filename,model.feature_importances_)

importances_df = pd.DataFrame()
importances_df['features'] = features
importances_df['importances'] = importances
importances_df.sort_values('feature_importances', ascending=False)



f, ax1 = plt.subplots(1, 1, sharex=True, figsize=(6, 4))

ax1.plot(a['P'], color="blue", linestyle="-", linewidth=1.5, label="Measurements")
ax1.plot(a['A'], color="green", linestyle="--", linewidth=1.5, label="Proposed model")

plt.legend(loc='upper right')
plt.xticks(fontsize=8,fontweight='normal')
plt.yticks(fontsize=8,fontweight='normal')
plt.title('Predictions for GBDT')
plt.xlabel('Sequence', fontsize=10)
plt.ylabel('Water depth_Ha_S2_30(m)_GBDT_depth_iceset-2_test_GBDT_0802', fontsize=10)
#plt.xlim(0, 25)
plt.savefig('ceshi/plots/S2_30m_ALL_GBDT_iceset-2_test_GBDT_0802.png', format='png')
plt.show()
