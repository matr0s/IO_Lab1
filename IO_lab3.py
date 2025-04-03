import time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ======= 1. Символьне визначення функції та автоматичне обчислення похідних =======

def define_function_and_derivatives():
    """Визначення функції та її похідних за допомогою sympy"""
    # Створюємо символьну змінну x
    x = sp.Symbol('x')
    
    # Визначаємо функцію (тут можна задавати іншу функцію за бажанням)
    # Мій варіант 16: f(x) = 2^(sin x) * 3^x - 8
    expr = 2**(sp.sin(x)) * 3**x - 8
    
    # Обчислюємо першу та другу похідні
    expr_diff = sp.diff(expr, x)
    expr_diff2 = sp.diff(expr_diff, x)
    
    # Перетворюємо вирази у функції Python
    f = sp.lambdify(x, expr, 'math')
    df = sp.lambdify(x, expr_diff, 'math')
    d2f = sp.lambdify(x, expr_diff2, 'math')
    
    # Вивід формул (для перевірки)
    print(f"Функція мого варіанту 16: f(x) = 2^(sin x) * 3^x - 8")
    print(f"Перша та друга похідні обчислені автоматично")
    print("-" * 60)
    
    return f, df, d2f

# ======= 2. Інтервальний метод Ньютона =======

def interval_newton_min(f, df, d2f, a, b, epsilon=1e-6, max_iter=100):
    """
    Інтервальний метод Ньютона для пошуку мінімуму функції
    Параметри:
    f - функція
    df - перша похідна
    d2f - друга похідна
    a, b - межі початкового інтервалу
    epsilon - точність (критерій зупинки)
    max_iter - максимальна кількість ітерацій
    
    Повертає:
    X - кінцевий інтервал, де знаходиться мінімум
    iterations - кількість виконаних ітерацій
    history - історія ітерацій
    """
    # Початковий інтервал
    X = [a, b]
    # Лічильник ітерацій та історія
    iterations = 0
    history = []
    # Перевіряємо, чи похідна змінює знак на інтервалі
    df_a = df(a)
    df_b = df(b)
    # Якщо похідна не змінює знак, мінімум може бути лише на межі інтервалу
    if df_a * df_b > 0:
        print("Увага: перша похідна не змінює знак на інтервалі.")
        print("Мінімум може бути лише на межі інтервалу.")
        # Перевіряємо, на якій межі мінімум
        if f(a) <= f(b):
            return [a, a], 0, []  # Мінімум у точці a
        else:
            return [b, b], 0, []  # Мінімум у точці b
    # Основний цикл методу
    for i in range(max_iter):
        iterations += 1
        # Знаходимо середину інтервалу
        m_X = (X[0] + X[1]) / 2
        # Обчислюємо значення першої похідної в середині
        df_m = df(m_X)
        # Знаходимо мінімальне значення другої похідної на інтервалі
        d2f_min = float('inf')
        d2f_max = float('-inf')
        # Розбиваємо інтервал на 20 точок для обчислення мінімуму другої похідної
        num_points = 20
        step = (X[1] - X[0]) / (num_points - 1) if num_points > 1 else 0

        for j in range(num_points):
            x_j = X[0] + j * step
            d2f_j = d2f(x_j)
            d2f_min = min(d2f_min, d2f_j)
            d2f_max = max(d2f_max, d2f_j)        
        # Перевіряємо умову існування мінімуму (друга похідна > 0)
        if d2f_min <= 0:
            print(f"Ітерація {iterations}: Увага! f''(x) ≤ 0 на деяких ділянках.")
            print("Функція може мати не мінімум, а максимум або точку перегину.")
        # Обчислюємо нове наближення за формулою Ньютона:
        # N = m - f'(m) / f''(m), але для гарантованої локалізації
        # використовуємо мінімальне значення f''(x)
        N = m_X - df_m / d2f_min
        # Знаходимо перетин з поточним інтервалом
        new_X = [max(X[0], min(N, X[1])), min(X[1], max(N, X[0]))]
        # Зберігаємо дані про ітерацію
        history.append({
            'iteration': iterations,
            'interval': X.copy(),
            'midpoint': m_X,
            'df_midpoint': df_m,
            'd2f_min': d2f_min,
            'd2f_max': d2f_max,
            'new_interval': new_X.copy(),
            'width': new_X[1] - new_X[0]
        })
        # Виводимо інформацію про ітерацію (з округленням)
        print(f"Ітерація {iterations}:")
        print(f"  Інтервал: [{X[0]:.4f}, {X[1]:.4f}], ширина: {X[1]-X[0]:.4f}")
        print(f"  Середина: {m_X:.4f}, f({m_X:.4f}) = {f(m_X):.4f}")
        print(f"  f'({m_X:.4f}) = {df_m:.4f}")
        print(f"  min f''(x) = {d2f_min:.4f}, max f''(x) = {d2f_max:.4f}")
        print(f"  Новий інтервал: [{new_X[0]:.4f}, {new_X[1]:.4f}], ширина: {new_X[1]-new_X[0]:.4f}")
        print()
        
        # Перевіряємо критерій збіжності (ширина інтервалу < epsilon)
        if abs(new_X[1] - new_X[0]) < epsilon:
            X = new_X
            break
        # Оновлюємо інтервал для наступної ітерації
        X = new_X
    # Якщо вийшли з циклу через досягнення максимальної кількості ітерацій
    if iterations == max_iter and abs(X[1] - X[0]) >= epsilon:
        print("Увага: досягнуто максимальну кількість ітерацій без досягнення потрібної точності.")
    return X, iterations, history

# ======= 3. Функції для візуалізації =======

def plot_function_and_result(f, a, b, result_interval, history=None):
    """Візуалізація функції та результату пошуку мінімуму"""
    # Створюємо фігуру
    plt.figure(figsize=(10, 6))
    # Точки для графіка (трохи розширюємо інтервал для кращої візуалізації)
    x_vals = np.linspace(a - (b-a)*0.1, b + (b-a)*0.1, 1000)
    y_vals = [f(x) for x in x_vals]
    # Будуємо графік функції
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
    # Виділяємо початковий інтервал
    plt.axvspan(a, b, alpha=0.1, color='gray', label='Початковий інтервал')
    # Виділяємо результуючий інтервал
    approx_min = (result_interval[0] + result_interval[1]) / 2
    plt.axvspan(result_interval[0], result_interval[1], alpha=0.3, color='r',
               label=f'Результуючий інтервал: [{result_interval[0]:.4f}, {result_interval[1]:.4f}]')
    # Позначаємо точку мінімуму
    plt.scatter([approx_min], [f(approx_min)], color='r', s=100, zorder=5,
               label=f'Мінімум: x ≈ {approx_min:.4f}, f(x) ≈ {f(approx_min):.4f}')
    # Якщо є історія ітерацій, показуємо процес збіжності
    if history and len(history) > 0:
        for i, step in enumerate(history):
            plt.axvspan(step['interval'][0], step['interval'][1], alpha=0.1, color='green')
            plt.scatter([step['midpoint']], [f(step['midpoint'])], color='green', s=30, zorder=4)
            plt.annotate(f"{i+1}", (step['midpoint'], f(step['midpoint'])), 
                         xytext=(step['midpoint'], f(step['midpoint']) - 0.2),
                         ha='center', fontsize=8)
    # Налаштування графіка
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Пошук мінімуму функції інтервальним методом Ньютона')
    plt.legend()
    # Зберігаємо та показуємо графік
    plt.savefig('newton_method_result.png')
    plt.tight_layout()
    plt.show()

def plot_derivatives(f, df, d2f, a, b, approx_min):
    """Візуалізація функції та її похідних"""
    # Створюємо фігуру
    plt.figure(figsize=(10, 8))
    # Точки для графіків
    x_vals = np.linspace(a - (b-a)*0.1, b + (b-a)*0.1, 1000)
    y_vals = [f(x) for x in x_vals]
    y_df_vals = [df(x) for x in x_vals]
    y_d2f_vals = [d2f(x) for x in x_vals]
    # Графік функції
    plt.subplot(3, 1, 1)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2)
    plt.axvline(x=approx_min, color='r', linestyle='--', alpha=0.7)
    plt.scatter([approx_min], [f(approx_min)], color='r', s=50)
    plt.grid(True)
    plt.ylabel('f(x)')
    plt.title('Функція та її похідні')
    # Графік першої похідної
    plt.subplot(3, 1, 2)
    plt.plot(x_vals, y_df_vals, 'g-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=approx_min, color='r', linestyle='--', alpha=0.7)
    plt.scatter([approx_min], [df(approx_min)], color='r', s=50)
    plt.grid(True)
    plt.ylabel('f\'(x)')
    # Графік другої похідної
    plt.subplot(3, 1, 3)
    plt.plot(x_vals, y_d2f_vals, 'r-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=approx_min, color='r', linestyle='--', alpha=0.7)
    plt.scatter([approx_min], [d2f(approx_min)], color='r', s=50)
    plt.grid(True)
    plt.ylabel('f\'\'(x)')
    plt.xlabel('x')
    # Зберігаємо та показуємо графік
    plt.savefig('derivatives.png')
    plt.tight_layout()
    plt.show()

# ======= 4. Основна функція =======

def main():
    """Основна функція програми"""
    # Параметри задачі
    a = 1.1
    b = 1.9
    epsilon = 1e-6    # Точність

    print("Інтервальний метод Ньютона для пошуку мінімуму функції")
    print("-" * 60)
    
    # Визначаємо функцію та її похідні
    f, df, d2f = define_function_and_derivatives()
    
    print(f"Пошук мінімуму на інтервалі [{a}, {b}]")
    print(f"Точність: {epsilon}")
    print("-" * 60)
    
    # Аналіз функції на інтервалі (з округленням)
    print(f"Значення функції на кінцях інтервалу:")
    print(f"f({a}) = {f(a):.4f}")
    print(f"f({b}) = {f(b):.4f}")
    print(f"Значення першої похідної:")
    print(f"f'({a}) = {df(a):.4f}")
    print(f"f'({b}) = {df(b):.4f}")
    print(f"Значення другої похідної:")
    print(f"f''({a}) = {d2f(a):.4f}")
    print(f"f''({b}) = {d2f(b):.4f}")
    print("-" * 60)
    
    # Перевірка умов існування мінімуму
    if df(a) * df(b) < 0:
        print("Перша похідна змінює знак на інтервалі - можливо існує внутрішня точка мінімуму.")
    else:
        print("Перша похідна не змінює знак на інтервалі - мінімум може бути лише на межі.")
    
    if d2f(a) > 0 and d2f(b) > 0:
        print("Друга похідна додатна на кінцях інтервалу - функція може мати мінімум.")
    else:
        print("Увага: друга похідна не є строго додатною на кінцях інтервалу.")
    print("-" * 60)
    
    # Вимірюємо час виконання
    start_time = time.time()
    
    # Застосовуємо інтервальний метод Ньютона
    print("Застосування інтервального методу Ньютона:")
    result_interval, iterations, history = interval_newton_min(f, df, d2f, a, b, epsilon)
    
    # Час виконання
    execution_time = time.time() - start_time
    
    # Виводимо результати (з округленням)
    print("-" * 60)
    print("Результати:")
    print(f"Початковий інтервал: [{a}, {b}]")
    print(f"Результуючий інтервал: [{result_interval[0]:.6f}, {result_interval[1]:.6f}]")
    print(f"Ширина результуючого інтервалу: {result_interval[1] - result_interval[0]:.6f}")
    
    # Оцінка точки мінімуму
    approx_min = (result_interval[0] + result_interval[1]) / 2
    print(f"Оцінка точки мінімуму: x ≈ {approx_min:.6f}")
    print(f"Значення функції в точці мінімуму: f({approx_min:.6f}) = {f(approx_min):.6f}")
    print(f"Значення першої похідної: f'({approx_min:.6f}) = {df(approx_min):.6f}")
    print(f"Значення другої похідної: f''({approx_min:.6f}) = {d2f(approx_min):.6f}")
    
    print(f"Кількість ітерацій: {iterations}")
    print(f"Час виконання: {execution_time:.6f} секунд")
    
    # Візуалізація результатів
    try:
        plot_function_and_result(f, a, b, result_interval, history)
        plot_derivatives(f, df, d2f, a, b, approx_min)
        print("\nГрафіки збережені у файлах 'newton_method_result.png' та 'derivatives.png'")
    except Exception as e:
        print(f"\nПомилка при візуалізації: {e}")
        print("Переконайтеся, що встановлено бібліотеку matplotlib: pip install matplotlib")

if __name__ == "__main__":
    main()