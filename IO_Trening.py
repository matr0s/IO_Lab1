import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

# 1. Визначення функції системи для варіанту №18
def system_function(x, a, b):
    """Функція системи для варіанту 18: f(x) = (x-a)/(x-0.75) + b*cos(x+0.38)"""
    return (x - a) / (x - 0.75) + b * np.cos(x + 0.38)

# 2. Генерація даних для тестування
def generate_data(x_range, true_a, true_b, error_percentage=20):
    """Генерує інтервальні дані вимірювань із заданою похибкою"""
    # Масиви для зберігання даних
    valid_x = []
    y_true = []
    
    # Обчислення значень функції, уникаючи точки x=0.75
    for x in x_range:
        try:
            if abs(x - 0.75) < 0.1:  # Уникаємо точок біля x=0.75
                continue
                
            y = system_function(x, true_a, true_b)
            y_true.append(y)
            valid_x.append(x)
        except:
            continue
    
    # Створення інтервалів з похибкою
    y_lower = [y * (1 - error_percentage/100) for y in y_true]
    y_upper = [y * (1 + error_percentage/100) for y in y_true]
    
    y_intervals = list(zip(y_lower, y_upper))
    
    return valid_x, y_true, y_intervals

# 3. Функція для обчислення відхилення (функція мети)
def objective_function(params, x_points, y_intervals):
    """Обчислює максимальне відхилення від інтервальних значень"""
    a, b = params
    max_deviation = 0
    
    for i, x in enumerate(x_points):
        try:
            # Обчислюємо значення функції з поточними параметрами
            y_calc = system_function(x, a, b)
            y_lower, y_upper = y_intervals[i]
            
            # Визначаємо відхилення від допустимого інтервалу
            if y_calc < y_lower:
                deviation = y_lower - y_calc
            elif y_calc > y_upper:
                deviation = y_calc - y_upper
            else:
                deviation = 0
                
            max_deviation = max(max_deviation, deviation)
        except:
            # У випадку помилки (наприклад, ділення на нуль)
            return float('inf')
    
    return max_deviation

# 4. Генерація випадкового вектора
def generate_random_vector(dimension):
    """Генерує випадковий вектор з рівномірним розподілом на [-1, 1]"""
    return [random.uniform(-1, 1) for _ in range(dimension)]

# 5. Основний алгоритм випадкового пошуку з використанням спрямованого конуса
def random_search(x_points, y_intervals, initial_params, radius=1.2, k_dec=0.8, 
                  max_iterations=1000, max_no_improvement=10, num_vectors=100):
    """
    Метод випадкового пошуку з використанням спрямованого конуса
    
    Параметри:
    x_points - точки вимірювання
    y_intervals - інтервальні значення функції
    initial_params - початкові значення параметрів [a0, b0]
    radius - початковий радіус пошуку (r)
    k_dec - коефіцієнт зменшення радіусу (kdec)
    max_iterations - максимальна кількість ітерацій
    max_no_improvement - максимальна кількість ітерацій без покращення (M)
    num_vectors - кількість випадкових векторів на кожній ітерації (N)
    """
    start_time = time.time()
    
    # Ініціалізація змінних
    current_params = initial_params.copy()
    best_params = initial_params.copy()
    best_objective = objective_function(best_params, x_points, y_intervals)
    
    no_improvement_count = 0  # m - лічильник ітерацій без покращення
    iterations = 0
    
    # Історія пошуку для аналізу
    history = {'a': [best_params[0]], 'b': [best_params[1]], 'objective': [best_objective]}
    
    while iterations < max_iterations and best_objective > 0:
        # Генеруємо N випадкових векторів
        trial_vectors = []
        
        for _ in range(num_vectors):
            # Генеруємо випадковий вектор
            random_vector = generate_random_vector(len(current_params))
            
            # Формуємо новий набір параметрів: bₖ = bₖ₋₁ + r · ξₖ
            trial_params = [current_params[i] + radius * random_vector[i] 
                           for i in range(len(current_params))]
            trial_vectors.append(trial_params)
        
        # Обчислюємо значення функції мети для всіх пробних векторів
        objective_values = []
        for trial_params in trial_vectors:
            objective_values.append(objective_function(trial_params, x_points, y_intervals))
        
        # Знаходимо найкращий вектор серед спроб
        best_trial_index = objective_values.index(min(objective_values))
        best_trial_params = trial_vectors[best_trial_index]
        best_trial_objective = objective_values[best_trial_index]
        
        # Якщо знайдено краще рішення: F(bᵢ) < F(b₀)
        if best_trial_objective < best_objective:
            best_params = best_trial_params.copy()
            best_objective = best_trial_objective
            current_params = best_trial_params.copy()
            no_improvement_count = 0
            
            # Зберігаємо історію
            history['a'].append(best_params[0])
            history['b'].append(best_params[1])
            history['objective'].append(best_objective)
        else:
            no_improvement_count += 1  # m = m + 1
        
        # Якщо довго немає покращення: m > M
        if no_improvement_count >= max_no_improvement:
            radius *= k_dec  # r = kdec · r
            no_improvement_count = 0  # m = 0
        
        iterations += 1
        
        # Логування
        if iterations % 10 == 0:
            print(f"Ітерація {iterations}: a = {best_params[0]:.6f}, b = {best_params[1]:.6f}, F(a,b) = {best_objective:.6f}")
        
        # Умови завершення пошуку
        if best_objective == 0 or radius < 1e-6:
            break
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return best_params, best_objective, iterations, execution_time, history

# 6. Візуалізація результатів
def visualize_results(x_points, y_true, y_lower, y_upper, best_params):
    """Візуалізує результати оптимізації"""
    a_opt, b_opt = best_params
    
    plt.figure(figsize=(12, 8))
    
    # Відображення інтервалів допустимих значень
    plt.fill_between(x_points, y_lower, y_upper, color='lightgray', alpha=0.5, 
                     label='Допустимі інтервали')
    
    # Відображення вимірюваних значень
    plt.plot(x_points, y_true, 'b-', label='Вимірювані значення')
    
    # Відображення оптимізованої функції
    x_smooth = np.linspace(min(x_points), max(x_points), 100)
    y_calculated = []
    valid_x_smooth = []
    
    for x in x_smooth:
        try:
            if abs(x - 0.75) < 0.1:
                continue
            y = system_function(x, a_opt, b_opt)
            y_calculated.append(y)
            valid_x_smooth.append(x)
        except:
            continue
    
    plt.plot(valid_x_smooth, y_calculated, 'r--', 
             label=f'Оптимізовані значення (a = {a_opt:.4f}, b = {b_opt:.4f})')
    
    # Оформлення графіка
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Результати оптимізації параметрів a і b')
    plt.grid(True)
    plt.savefig('optimization_results.png')
    plt.show()

# 7. Візуалізація історії пошуку
def visualize_history(history):
    """Візуалізує історію зміни параметрів та функції мети"""
    plt.figure(figsize=(15, 5))
    
    # Графік історії параметра a
    plt.subplot(1, 3, 1)
    plt.plot(history['a'])
    plt.xlabel('Ітерація')
    plt.ylabel('Значення параметра a')
    plt.title('Історія параметра a')
    plt.grid(True)
    
    # Графік історії параметра b
    plt.subplot(1, 3, 2)
    plt.plot(history['b'])
    plt.xlabel('Ітерація')
    plt.ylabel('Значення параметра b')
    plt.title('Історія параметра b')
    plt.grid(True)
    
    # Графік історії функції мети (в логарифмічному масштабі)
    plt.subplot(1, 3, 3)
    plt.plot(history['objective'])
    plt.xlabel('Ітерація')
    plt.ylabel('Значення функції мети')
    plt.title('Історія функції мети')
    plt.yscale('log')  # Логарифмічний масштаб для наочності
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('search_history.png')
    plt.show()

# 8. Головна функція
def main():
    print("Оптимізація параметрів функції f(x) = (x-a)/(x-0.75) + b*cos(x+0.38)")
    print("Метод випадкового пошуку за алгоритмом з лекції")
    
    # Задаємо початкові параметри
    true_a = 1.0  # Модельні значення для генерації даних
    true_b = 2.0
    
    # Створюємо діапазон точок x для вимірювання, уникаючи точку x=0.75
    x_range = np.concatenate([np.linspace(-2, 0.6, 20), np.linspace(0.9, 4, 20)])
    
    # Генеруємо дані вимірювань
    x_points, y_true, y_intervals = generate_data(x_range, true_a, true_b)
    y_lower = [interval[0] for interval in y_intervals]
    y_upper = [interval[1] for interval in y_intervals]
    
    # Задаємо параметри алгоритму відповідно до прикладу з лекції
    initial_params = [-0.2, 2.0]  # Початкові значення [a0, b0]
    radius = 1.2          # Початковий радіус пошуку (r)
    k_dec = 0.8           # Коефіцієнт зменшення радіуса (kdec)
    max_iterations = 1000  # Максимальна кількість ітерацій
    max_no_improvement = 10  # M - кількість ітерацій без покращення
    num_vectors = 100     # N - кількість випадкових векторів (рекомендовано не менше 100)
    
    # Запускаємо алгоритм оптимізації
    best_params, best_objective, iterations, execution_time, history = random_search(
        x_points, y_intervals, initial_params, radius, k_dec, max_iterations, 
        max_no_improvement, num_vectors
    )
    
    # Виводимо результати
    print(f"\nРезультати оптимізації:")
    print(f"Оптимальні значення: a = {best_params[0]:.6f}, b = {best_params[1]:.6f}")
    print(f"Значення функції мети = {best_objective:.6f}")
    print(f"Кількість ітерацій = {iterations}")
    print(f"Час виконання = {execution_time:.6f} секунд")
    
    # Візуалізуємо результати
    visualize_results(x_points, y_true, y_lower, y_upper, best_params)
    
    # Візуалізуємо історію пошуку
    visualize_history(history)

# Запускаємо програму
if __name__ == "__main__":
    main()