import time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ======= 1. Символьне визначення функції та автоматичне обчислення похідних =======

def define_function_and_derivatives():
    """Визначення функції та її похідної за допомогою sympy"""
    # Створюємо символьну змінну x
    x = sp.Symbol('x') 
    # Визначаємо функцію (тут можна задавати іншу функцію за бажанням)
    # Мій варіант 16: f(x) = 2^(sin x) * 3^x - 8
    expr = 2**(sp.sin(x)) * 3**x - 8
    # Обчислюємо першу похідну
    expr_diff = sp.diff(expr, x)
    # Перетворюємо вирази у функції Python
    f = sp.lambdify(x, expr, 'math')
    df = sp.lambdify(x, expr_diff, 'math')
    # Вивід формул (для перевірки)
    print(f"Функція мого варіанту 16: f(x) = 2^(sin x) * 3^x - 8")
    print(f"Перша похідна: {expr_diff}")
    print("-" * 60)
    return f, df

# ======= 2. Метод Ньютона для пошуку кореня функції (де f(x) = 0) ======

def interval_newton_root(f, df, a, b, epsilon=1e-6, max_iter=100):
    #Інтервальний метод Ньютона для пошуку кореня функції f(x) = 0 на інтервалі [a, b]
    # Перевіряємо, чи існує корінь на інтервалі за теоремою Больцано
    if f(a) * f(b) > 0:
        print("Увага: функція не змінює знак на інтервалі, корінь може не існувати.")
        print("Будемо шукати точку, де значення функції найближче до нуля.")
    # Початковий інтервал
    X = [a, b]
    # Лічильник ітерацій та історія
    iterations = 0
    history = []
    # Основний цикл методу
    for i in range(max_iter):
        iterations += 1       
        # Знаходимо середину інтервалу
        m_X = (X[0] + X[1]) / 2
        # Обчислюємо значення функції та її похідної в середині
        f_m = f(m_X)
        df_m = df(m_X)
        # Перевіряємо критерій збіжності за значенням функції
        if abs(f_m) < epsilon:
            print(f"Знайдено корінь функції: f({m_X:.6f}) ≈ 0")
            break
        # Перевіряємо можливість застосування методу Ньютона
        if abs(df_m) < epsilon:
            print("Похідна близька до нуля, метод може розбігатися.")
            break
        # Обчислюємо наступну точку за формулою Ньютона
        N = m_X - f_m / df_m
        # Обмежуємо точку Ньютона інтервалом [a, b]
        N = max(a, min(b, N))
        # Обчислюємо значення функції в точці Ньютона
        f_N = f(N)
        # Оновлюємо інтервал відповідно до знаку функції (спрощена логіка)
        if f_m * f_N <= 0:  # Функція змінює знак між m_X і N
            new_X = [min(m_X, N), max(m_X, N)]
        elif f(X[0]) * f_N <= 0:  # Функція змінює знак між X[0] і N
            new_X = [X[0], N]
        else:  # Функція змінює знак між N і X[1]
            new_X = [N, X[1]]
        # Зберігаємо дані про ітерацію
        history.append({
            'iteration': iterations,
            'interval': X.copy(),
            'midpoint': m_X,
            'f_midpoint': f_m,
            'df_midpoint': df_m,
            'newton_point': N,
            'f_newton_point': f_N,
            'new_interval': new_X.copy(),
            'width': new_X[1] - new_X[0]
        })
        # Виводимо інформацію про ітерацію
        print(f"Ітерація {iterations}:")
        print(f"  Інтервал: [{X[0]:.6f}, {X[1]:.6f}], ширина: {X[1]-X[0]:.6f}")
        print(f"  Середина: {m_X:.6f}, f({m_X:.6f}) = {f_m:.6f}")
        print(f"  f'({m_X:.6f}) = {df_m:.6f}")
        print(f"  Точка Ньютона: {N:.6f}, f({N:.6f}) = {f_N:.6f}")
        print(f"  Новий інтервал: [{new_X[0]:.6f}, {new_X[1]:.6f}], ширина: {new_X[1]-new_X[0]:.6f}")
        print()
        # Об'єднаний критерій зупинки: досягнута точність або занадто малий інтервал
        if abs(new_X[1] - new_X[0]) < epsilon:
            X = new_X
            break
        # Оновлюємо інтервал для наступної ітерації
        X = new_X
    # Якщо вийшли з циклу без досягнення точності
    if iterations == max_iter:
        print("Увага: досягнуто максимальну кількість ітерацій без досягнення потрібної точності.")
    return X, iterations, history

# ======= 3. Функції для візуалізації =======

def plot_function_and_result(f, a, b, result_interval, history=None):
    """Візуалізація функції та результату пошуку кореня"""
    # Створюємо фігуру
    plt.figure(figsize=(10, 6))
    # Точки для графіка
    x_vals = np.linspace(a - (b-a)*0.1, b + (b-a)*0.1, 1000)
    y_vals = [f(x) for x in x_vals]
    # Обчислюємо корінь як середину результуючого інтервалу
    approx_root = (result_interval[0] + result_interval[1]) / 2
    # Будуємо графік
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvspan(a, b, alpha=0.1, color='gray', label='Початковий інтервал')
    plt.axvspan(result_interval[0], result_interval[1], alpha=0.3, color='r',
               label=f'Результуючий інтервал: [{result_interval[0]:.4f}, {result_interval[1]:.4f}]')
    plt.scatter([approx_root], [f(approx_root)], color='r', s=100, zorder=5,
               label=f'Корінь: x ≈ {approx_root:.4f}, f(x) ≈ {f(approx_root):.4f}')
    # Відображаємо історію ітерацій, якщо вона надана
    if history and len(history) > 0:
        for i, step in enumerate(history):
            plt.axvspan(step['interval'][0], step['interval'][1], alpha=0.1, color='green')
            plt.scatter([step['midpoint']], [step['f_midpoint']], color='green', s=30, zorder=4)
            plt.annotate(f"{i+1}", (step['midpoint'], step['f_midpoint']), 
                         xytext=(step['midpoint'], step['f_midpoint'] - 0.2),
                         ha='center', fontsize=8)  
    # Налаштування графіка
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Пошук кореня функції інтервальним методом Ньютона')
    plt.legend()
    plt.tight_layout()
    # Зберігаємо та показуємо графік
    plt.savefig('newton_method_result.png')
    plt.show()

def plot_function_and_derivative(f, df, a, b, approx_root):
    """Візуалізація функції та її похідної"""
    # Створюємо фігуру
    plt.figure(figsize=(10, 6))  
    # Точки для графіків
    x_vals = np.linspace(a - (b-a)*0.1, b + (b-a)*0.1, 1000)
    y_vals = [f(x) for x in x_vals]
    y_df_vals = [df(x) for x in x_vals]
    # Графік функції
    plt.subplot(2, 1, 1)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=approx_root, color='r', linestyle='--', alpha=0.7)
    plt.scatter([approx_root], [f(approx_root)], color='r', s=50)
    plt.grid(True)
    plt.ylabel('f(x)')
    plt.title('Функція та її похідна')
    # Графік першої похідної
    plt.subplot(2, 1, 2)
    plt.plot(x_vals, y_df_vals, 'g-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=approx_root, color='r', linestyle='--', alpha=0.7)
    plt.scatter([approx_root], [df(approx_root)], color='r', s=50)
    plt.grid(True)
    plt.ylabel('f\'(x)')
    plt.xlabel('x')
    # Зберігаємо та показуємо графік
    plt.tight_layout()
    plt.savefig('derivative.png')
    plt.show()

# ======= 4. Основна функція =======

def main():
    """Основна функція програми"""
    # Параметри задачі
    a = 1.1
    b = 1.9
    epsilon = 1e-6    # Точність
    print("Інтервальний метод Ньютона для пошуку кореня функції")
    print("-" * 60)
    # Визначаємо функцію та її похідну
    f, df = define_function_and_derivatives()
    print(f"Пошук кореня на інтервалі [{a}, {b}]")
    print(f"Точність: {epsilon}")
    print("-" * 60)
    # Аналіз функції на інтервалі
    print(f"Значення функції на кінцях інтервалу:")
    print(f"f({a}) = {f(a):.4f}")
    print(f"f({b}) = {f(b):.4f}")
    print(f"Значення першої похідної:")
    print(f"f'({a}) = {df(a):.4f}")
    print(f"f'({b}) = {df(b):.4f}")
    print("-" * 60)
    # Аналізуємо функцію на інтервалі
    print("Аналіз функції на інтервалі:")
    if f(a) * f(b) < 0:
        print("Функція змінює знак на інтервалі - існує корінь.")
    else:
        print("Функція не змінює знак на інтервалі - спробуємо знайти точку, де функція найближча до нуля.")
    print("-" * 60)
    # Застосовуємо інтервальний метод Ньютона
    print("Застосування інтервального методу Ньютона:")
    result_interval, iterations, history = interval_newton_root(f, df, a, b, epsilon)
    # Виводимо результати
    print("-" * 60)
    print("Результати:")
    print(f"Початковий інтервал: [{a}, {b}]")
    print(f"Результуючий інтервал: [{result_interval[0]:.6f}, {result_interval[1]:.6f}]")
    print(f"Ширина результуючого інтервалу: {result_interval[1] - result_interval[0]:.6f}")
    # Оцінка точки кореня
    approx_root = (result_interval[0] + result_interval[1]) / 2
    print(f"Оцінка точки кореня: x ≈ {approx_root:.6f}")
    print(f"Значення функції в точці: f({approx_root:.6f}) = {f(approx_root):.6f}")
    print(f"Значення першої похідної: f'({approx_root:.6f}) = {df(approx_root):.6f}")    
    print(f"Кількість ітерацій: {iterations}")
    # Візуалізація результатів
    try:
        plot_function_and_result(f, a, b, result_interval, history)
        plot_function_and_derivative(f, df, a, b, approx_root)
        print("\nГрафіки збережені у файлах 'newton_method_result.png' та 'derivative.png'")
    except Exception as e:
        print(f"\nПомилка при візуалізації: {e}")
        print("Переконайтеся, що встановлено бібліотеку matplotlib: pip install matplotlib")

if __name__ == "__main__":
    main()