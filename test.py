
from service.client import BertClient

bc = BertClient()
s = ['你好。', '嗨！', '今日我本来打算去公园。']
a = bc.encode(s)
# a = bc.encode(['Today is good.'])
print(a.shape)
# print(s.type)
print(a[0][1].shape)
print(a[0][1])
print(a[0][4])
print(a[1][3])
print(a[0][5])
