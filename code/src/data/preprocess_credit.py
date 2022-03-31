from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

guest_csv = pd.read_csv('/data/default_credit_hetero_guest.csv')
host_csv = pd.read_csv('/data/default_credit_hetero_host.csv')

guest_numpy = guest_csv.to_numpy()
host_numpy = host_csv.to_numpy()
print(guest_numpy[0], host_numpy[0])

ss = MinMaxScaler()
ss1 = StandardScaler()

guest_numpy[:, 2:] = ss.fit_transform(guest_numpy[:, 2:])
host_numpy[:, 1:] = ss.fit_transform(host_numpy[:, 1:])

guest_numpy[:, 2:] = ss1.fit_transform(guest_numpy[:, 2:])
host_numpy[:, 1:] = ss1.fit_transform(host_numpy[:, 1:])


pd.DataFrame(guest_numpy).to_csv("/data/scaled_credit_hetero_guest.csv", index=False)
pd.DataFrame(host_numpy).to_csv("/data/scaled_credit_hetero_host.csv", index=False)
