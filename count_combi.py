def invert_combi(n, i):
    x = 0
    y = 0
    z = 0

    n -= 1

    while i >= (n - x) * (n - 1 - x) // 2:
        i -= ((n - x) * (n - 1 - x)) // 2
        x += 1

    # print(x, "left", i)
    n -= x

    # solve for (n-x)C2
    y = 0
    n -= 1

    while i >= (n - y):
        i -= n - y
        y += 1

    y += x + 1
    z = y + i + 1
    return x, y, z


def combi(n):
    count = 0
    for x in range(n):
        for y in range(x + 1, n):
            for z in range(y + 1, n):
                print(count, ":", x, y, z)
                assert (x, y, z) == invert_combi(n, count)
                count += 1


def main():
    combi(20)


if __name__ == "__main__":
    main()
