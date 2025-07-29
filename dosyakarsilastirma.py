with open("requirements.txt", "r") as f, open("requirements2.txt", "r") as f2:
    l1 = f.readlines()
    l2 = f2.readlines()
    print(*(set(l1) - set(l2)),file=open("fark.txt","w"))