import numpy as np

DEG_PER_RADIAN = 1 / np.pi


def eucdist(p, q=0):
    return np.sqrt(np.sum((p - q) ** 2))


def calculate_angle(u, v):
    mod_u = eucdist(u)
    mod_v = eucdist(v)
    num = eucdist(np.array([u[0] * mod_v - v[0] * mod_u, u[1] * mod_v - v[1] * mod_u]))
    den = eucdist(np.array([u[0] * mod_v + v[0] * mod_u, u[1] * mod_v + v[1] * mod_u]))
    return np.arctan2(num, den)


class Triangle:
    def __init__(self, A, B, C):
        A = np.array(A)
        B = np.array(B)
        C = np.array(C)
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)

        self.ab = eucdist(self.A, self.B)
        self.bc = eucdist(self.B, self.C)
        self.ca = eucdist(self.C, self.A)

        self.At = calculate_angle(A - C, B - A)
        self.Bt = calculate_angle(B - A, C - B)
        self.Ct = calculate_angle(C - B, A - C)

    def __str__(self):
        return "A {}, {:.2f}; B {}, {:.2f}; C {}, {:.2f};".format(
            self.A,
            self.At * DEG_PER_RADIAN,
            self.B,
            self.Bt * DEG_PER_RADIAN,
            self.C,
            self.Ct * DEG_PER_RADIAN,
        )

    def compare(self, other, delta=0.01):
        angles1 = np.array([self.At, self.Bt, self.Ct])
        sides1 = np.array([self.bc, self.ca, self.ab])

        angles2 = np.array(
            [
                [other.At, other.Bt, other.Ct],
                [other.At, other.Ct, other.Bt],
                [other.Bt, other.At, other.Ct],
                [other.Bt, other.Ct, other.At],
                [other.Ct, other.At, other.Bt],
                [other.Ct, other.Bt, other.At],
            ]
        )
        sides2 = np.array(
            [
                [other.bc, other.ca, other.ab],
                [other.bc, other.ab, other.ca],
                [other.ca, other.bc, other.ab],
                [other.ca, other.ab, other.bc],
                [other.ab, other.bc, other.ca],
                [other.ab, other.ca, other.bc],
            ]
        )
        diffs = np.zeros(6)
        ratios = np.zeros(6)
        for i in range(6):
            diffs[i] = eucdist(angles1, angles2[i, :])
            ratios[i] = np.max(sides1 / sides2[i, :])
        print(diffs)
        print(ratios[diffs <= delta])
        return diffs <= delta


def main():
    t1 = Triangle([-3, 0], [3, 0], [0, 4])
    t2 = Triangle([6, 0], [0, 8], [-6, 0])
    print(t1)
    print(t2)

    print(t1.compare(t2))
    print(t2.compare(t1))


if __name__ == "__main__":
    main()
