import random
import math
import matplotlib.pyplot as plt

L = 4  # длина хромосомы


def f(x: float) -> float:
    """Целевая функция варианта 16."""
    return math.sin(3 * math.pi * x) + 0.2 * x * x


def decode(bits: str) -> float:
    """Декодирование бинарной хромосомы в вещественное x на [0, 1]."""
    return int(bits, 2) / 15


def random_individual() -> str:
    """Случайная бинарная строка длины L."""
    return "".join(random.choice("01") for _ in range(L))


def fitness(bits: str) -> float:
    """Значение целевой функции для хромосомы."""
    return f(decode(bits))


def selection(population):
    """Селекция методом рулетки."""
    fits = [fitness(ind) for ind in population]
    s = sum(fits)

    if s == 0:
        return random.choice(population)

    r = random.uniform(0, s)
    acc = 0.0
    for ind, fit_val in zip(population, fits):
        acc += fit_val
        if acc >= r:
            return ind

    return population[-1]


def crossover(a: str, b: str, pc: float):
    """Одноточечный кроссовер."""
    if random.random() < pc:
        point = random.randint(1, L - 1)
        child1 = a[:point] + b[point:]
        child2 = b[:point] + a[point:]
        return child1, child2
    return a, b


def mutate(bits: str, pm: float) -> str:
    """Побитовая мутация."""
    bits_list = list(bits)
    for i in range(L):
        if random.random() < pm:
            bits_list[i] = "1" if bits_list[i] == "0" else "0"
    return "".join(bits_list)


def run_ga(N: int, pc: float, pm: float, G: int):
    """Запуск генетического алгоритма, возврат лучшего индивида и истории максимумов."""
    population = [random_individual() for _ in range(N)]
    best_hist = []

    for _ in range(G):
        fits = [fitness(ind) for ind in population]
        best_hist.append(max(fits))

        new_population = []
        for _ in range(N // 2):
            p1 = selection(population)
            p2 = selection(population)
            c1, c2 = crossover(p1, p2, pc)
            c1 = mutate(c1, pm)
            c2 = mutate(c2, pm)
            new_population.extend([c1, c2])

        population = new_population

    best_ind = max(population, key=fitness)
    return best_ind, decode(best_ind), fitness(best_ind), best_hist


def main():
    experiments = [
        ("Эксперимент 1", 4, 0.7, 0.1, 30),
        ("Эксперимент 2", 4, 0.3, 0.1, 30),
        ("Эксперимент 3", 4, 0.9, 0.2, 30),
    ]

    all_hist = []

    for name, N, pc, pm, G in experiments:
        best_bits, x_opt, f_opt, hist = run_ga(N, pc, pm, G)
        all_hist.append((name, hist))
        print(f"{name}:")
        print(f"Лучший: {best_bits} -> x={x_opt:.4f}, f={f_opt:.6f}\n")

    # График изменения максимального fitness
    plt.figure()
    for name, hist in all_hist:
        plt.plot(hist, label=name)
    plt.xlabel("Поколение")
    plt.ylabel("Максимальный fitness")
    plt.legend()
    plt.grid(True)
    plt.title("Сходимость генетического алгоритма")
    plt.show()


if __name__ == "__main__":
    main()
