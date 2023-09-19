hexMap = {'坤卦': (0, 0, 0), '艮卦': (1, 0, 0), '坎卦': (0, 1, 0), '巽卦': (1, 1, 0), '中卦': (0.5, 0.5, 0.5),
          '震卦': (0, 0, 1), '離卦': (1, 0, 1), '兌卦': (0, 1, 1), '乾卦': (1, 1, 1)}
D = 0.5
time = 1
length = 1
epochs = 10
for name in list(hexMap.keys()):
    env = list(hexMap[name])
    print(name, env)
    for i in range(epochs):
        env = [env[0]] + env + [env[-1]]
        newEnv = env.copy()
        for j in range(1, len(env) - 1):
            newEnv[j] += (env[j - 1] + env[j + 1] - 2 * env[j]) * D * time / length ** 2
            if newEnv[j] > 1:
                newEnv[j] = 1
                print("yang overflow")
            if newEnv[j] < 0:
                newEnv[j] = 0
                print("yin overflow")
        env = newEnv[1:-1]
        print(env)