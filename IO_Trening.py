import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Визначення системної функції варіанту №18
def system_function(x, a, b):
    """
    Обчислює значення функції f(x) = (x-a)/(x-0.75) + b*cos(x+0.38)
    Параметри:
        x (float): Вхідне значення x
        a (float): Параметр a
        b (float): Параметр b
    Повертає:
        float: Значення функції
    Викидає:
        ValueError: Якщо x занадто близько до точки розриву 0.75
    """
    # Перевірка близькості до точки розриву
    if abs(x - 0.75) < 0.15:
        raise ValueError(f"Точка x={x} занадто близько до точки розриву x=0.75")
    
    return (x - a) / (x - 0.75) + b * np.cos(x + 0.38)

def generate_data(x_range, true_a, true_b, error_percentage=20):
    """
    Генерує інтервальні дані вимірювань із заданою похибкою
    Параметри:
        x_range (array): Масив точок x для генерації даних
        true_a (float): Істинне значення параметра a
        true_b (float): Істинне значення параметра b
        error_percentage (float): Відсоток похибки для створення інтервалів
    Повертає:
        tuple: (дійсні x, точні значення y, інтервали y)
    """
    valid_x = []
    y_true = []
    
    # Обчислення значень функції, уникаючи точки розриву
    for x in x_range:
        try:
            if abs(x - 0.75) < 0.15:
                continue
                
            y = system_function(x, true_a, true_b)
            y_true.append(y)
            valid_x.append(x)
        except Exception:
            continue
    
    # Створення інтервалів із заданою похибкою
    y_lower = [y * (1 - error_percentage/100) for y in y_true]
    y_upper = [y * (1 + error_percentage/100) for y in y_true]
    
    return np.array(valid_x), np.array(y_true), list(zip(y_lower, y_upper))

def objective_function(params, x_points, y_intervals):
    """
    Обчислює максимальне відхилення параметрів від інтервальних вимірювань
    
    Параметри:
        params (list): Параметри [a, b]
        x_points (array): Точки вимірювань x
        y_intervals (list): Інтервали вимірювань y у вигляді [(y_min_1, y_max_1), ...]
        
    Повертає:
        float: Максимальне відхилення від інтервалів
    """
    a, b = params
    max_deviation = 0
    deviations = []
    
    for i, x in enumerate(x_points):
        try:
            # Захист від точки розриву
            if abs(x - 0.75) < 0.15:
                continue
                
            # Обчислення значення функції з поточними параметрами
            y_calc = system_function(x, a, b)
            y_lower, y_upper = y_intervals[i]
            
            # Визначення відхилення від допустимого інтервалу
            if y_calc < y_lower:
                deviation = y_lower - y_calc
            elif y_calc > y_upper:
                deviation = y_calc - y_upper
            else:
                deviation = 0
            
            deviations.append(deviation)
            max_deviation = max(max_deviation, deviation)
        except Exception:
            # У випадку помилки (наприклад, ділення на нуль)
            return float('inf')
    
    # Перевірка чи є відхилення (якщо немає точок для порівняння)
    if not deviations:
        return float('inf')
    
    return max_deviation

def random_search(x_points, y_intervals, initial_params, radius=1.2, k_dec=0.8, 
                  max_iterations=1000, max_no_improvement=10, num_vectors=100):
    """
    Метод випадкового пошуку з використанням спрямованого конуса
    
    Параметри:
        x_points: точки вимірювання
        y_intervals: інтервальні значення функції
        initial_params: початкові значення параметрів [a0, b0]
        radius: початковий радіус пошуку (r)
        k_dec: коефіцієнт зменшення радіусу (kdec)
        max_iterations: максимальна кількість ітерацій
        max_no_improvement: максимальна кількість ітерацій без покращення (M)
        num_vectors: кількість випадкових векторів на кожній ітерації (N)
        
    Повертає:
        tuple: (оптимальні параметри, значення функції мети, кількість ітерацій, 
               час виконання, історія пошуку)
    """
    start_time = time.time()
    
    # Ініціалізація змінних
    current_params = initial_params.copy()
    best_params = initial_params.copy()
    best_objective = objective_function(best_params, x_points, y_intervals)
    
    print(f"Початкові параметри: a = {initial_params[0]:.4f}, b = {initial_params[1]:.4f}")
    print(f"Початкове значення функції мети: {best_objective:.6f}")
    
    no_improvement_count = 0
    iterations = 0
    
    # Історія пошуку для аналізу
    history = {'a': [best_params[0]], 'b': [best_params[1]], 'objective': [best_objective]}
    
    # Забезпечуємо мінімальну кількість ітерацій
    min_iterations = 10
    
    while (iterations < max_iterations and best_objective > 1e-10) or iterations < min_iterations:
        # Генеруємо N випадкових векторів
        trial_vectors = []
        
        for _ in range(num_vectors):
            # Випадковий вектор з рівномірним розподілом на [-1, 1]
            random_vector = [random.uniform(-1, 1) for _ in range(len(current_params))]
            
            # Формуємо новий набір параметрів: b_k = b_(k-1) + r * ξ_k
            trial_params = [current_params[i] + radius * random_vector[i] 
                           for i in range(len(current_params))]
            trial_vectors.append(trial_params)
        
        # Обчислюємо значення функції мети для всіх пробних векторів
        objective_values = [objective_function(params, x_points, y_intervals) 
                           for params in trial_vectors]
        
        # Знаходимо найкращий вектор серед спроб
        best_trial_index = objective_values.index(min(objective_values))
        best_trial_params = trial_vectors[best_trial_index]
        best_trial_objective = objective_values[best_trial_index]
        
            # Якщо знайдено краще рішення
        if best_trial_objective < best_objective:
            best_params = best_trial_params.copy()
            best_objective = best_trial_objective
            current_params = best_trial_params.copy()
            no_improvement_count = 0
            
            # Зберігаємо історію
            history['a'].append(best_params[0])
            history['b'].append(best_params[1])
            history['objective'].append(best_objective)
            
            # Виводимо інформацію тільки про значні покращення (більше 5%)
            if iterations == 0 or best_objective < history['objective'][-2] * 0.95:
                print(f"Ітерація {iterations}: a = {best_params[0]:.4f}, b = {best_params[1]:.4f}, "
                      f"F(a,b) = {best_objective:.6f}")
        else:
            no_improvement_count += 1
        
        # Якщо довго немає покращення
        if no_improvement_count >= max_no_improvement:
            old_radius = radius
            radius *= k_dec
            no_improvement_count = 0
        
        iterations += 1
        
        # Умови завершення пошуку
        if (best_objective <= 1e-10 and iterations >= min_iterations) or radius < 1e-6:
            print(f"Пошук завершено: функція мети = {best_objective:.8f}, радіус = {radius:.6f}")
            break
    
    execution_time = time.time() - start_time
    
    return best_params, best_objective, iterations, execution_time, history

def visualize_results(x_points, y_points, y_lower, y_upper, best_params, 
                      x_points_all=None, y_points_all=None, history=None):
    """
    Візуалізує результати оптимізації
    
    Параметри:
        x_points, y_points: експериментальні дані для оптимізації
        y_lower, y_upper: нижня та верхня межі інтервалів
        best_params: оптимальні значення параметрів
        x_points_all, y_points_all: всі експериментальні дані (опціонально)
        history: історія оптимізації (опціонально)
    """
    a_opt, b_opt = best_params
    
    # Графік 1: Результати оптимізації (x < 0.75)
    plt.figure(figsize=(12, 8))
    
    # Відображення інтервалів допустимих значень
    plt.fill_between(x_points, y_lower, y_upper, color='lightgray', alpha=0.5, 
                     label='Допустимі інтервали')
    
    # Відображення експериментальних даних
    plt.plot(x_points, y_points, 'bo-', markersize=6, 
             label='Експериментальні дані (для оптимізації)')
    
    # Якщо є повний набір точок, відображаємо їх
    if x_points_all is not None and y_points_all is not None:
        mask = np.array([x not in x_points for x in x_points_all])
        if any(mask):
            plt.plot(x_points_all[mask], y_points_all[mask], 'go', markersize=4, 
                     label='Експериментальні дані (не використані)')
    
    # Створюємо точки для гладкого відображення функції (до розриву)
    x_left = np.linspace(min(x_points), 0.7, 200)
    
    # Обчислюємо значення функції для лівої частини
    y_left = []
    valid_x_left = []
    for x in x_left:
        try:
            if abs(x - 0.75) < 0.2:
                continue
            y = system_function(x, a_opt, b_opt)
            y_left.append(y)
            valid_x_left.append(x)
        except Exception:
            continue
    
    # Оптимізована функція
    plt.plot(valid_x_left, y_left, 'r-', linewidth=2, 
             label=f'Оптимізована функція (a = {a_opt:.4f}, b = {b_opt:.4f})')
    
    # Позначаємо точку розриву
    plt.axvline(x=0.75, color='purple', linestyle='--', label='Точка розриву (x=0.75)')
    
    # Позначаємо точку x=0 та її значення
    try:
        y_at_zero = system_function(0, a_opt, b_opt)
        plt.plot(0, y_at_zero, 'ro', markersize=8)
        plt.annotate(f'f(0) = {y_at_zero:.4f}', 
                     xy=(0, y_at_zero), 
                     xytext=(0.5, y_at_zero + 1.0),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=10)
    except Exception as e:
        print(f"Помилка при обчисленні f(0): {e}")
    
    # Оформлення графіка
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Результати оптимізації параметрів a і b (тільки для x < 0.75)')
    plt.grid(True)
    plt.ylim(-10, 10)  # Обмежуємо діапазон y для кращої наочності
    plt.savefig('optimization_results.png')
    plt.show()
    
    # Графік 2: Екстраполяція функції на всі експериментальні точки
    if x_points_all is not None and y_points_all is not None:
        plt.figure(figsize=(15, 6))
        
        # Відображаємо всі експериментальні точки
        plt.scatter(x_points_all, y_points_all, color='blue', s=50, 
                    label='Всі експериментальні точки')
        
        # Позначаємо точки, використані для оптимізації
        plt.scatter(x_points, y_points, color='red', s=30, marker='x', 
                   label='Точки для оптимізації')
        
        # Створюємо щільну сітку для відображення функції
        x_dense_left = np.linspace(min(x_points_all), 0.7, 200)
        x_dense_right = np.linspace(0.8, max(x_points_all), 200)
        
        # Обчислюємо значення функції для лівої та правої частини
        y_left, valid_x_left = get_function_values(x_dense_left, a_opt, b_opt)
        y_right, valid_x_right = get_function_values(x_dense_right, a_opt, b_opt)
        
        # Відображаємо функцію
        plt.plot(valid_x_left, y_left, 'r-', label='Оптимізована функція (x < 0.75)')
        plt.plot(valid_x_right, y_right, 'r--', label='Екстраполяція функції (x > 0.75)')
        
        # Позначаємо точку розриву
        plt.axvline(x=0.75, color='purple', linestyle='--', label='Точка розриву (x=0.75)')
        
        # Оформлення графіка
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Порівняння екстраполяції функції на всі експериментальні точки')
        plt.grid(True)
        plt.ylim(-10, 10)
        plt.savefig('extrapolation.png')
        plt.show()
    
    # Якщо передана історія оптимізації, відображаємо її
    if history is not None:
        visualize_history(history)

def get_function_values(x_range, a, b):
    """
    Обчислює значення функції для масиву точок, уникаючи точку розриву
    
    Параметри:
        x_range: масив точок x
        a, b: параметри функції
        
    Повертає:
        tuple: (значення y, відповідні x)
    """
    y_values = []
    valid_x = []
    for x in x_range:
        try:
            if abs(x - 0.75) < 0.2:
                continue
            y = system_function(x, a, b)
            y_values.append(y)
            valid_x.append(x)
        except Exception:
            continue
    
    return y_values, valid_x

def visualize_history(history):
    """
    Візуалізує історію зміни параметрів та функції мети
    
    Параметри:
        history: історія оптимізації у вигляді словника
    """
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

def main():
    """
    Головна функція програми
    """
    print("Оптимізація параметрів функції f(x) = (x-a)/(x-0.75) + b*cos(x+0.38)")
    print("Метод випадкового пошуку з використанням спрямованого конуса")
    
    # Експериментальні дані (з графіка)
    x_points_all = np.array([-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14])
    y_points_all = np.array([3, 3, -3, 2, 4, -2.5, 0.5, 4, -2, 0, 5, -1, -1, 5, 0])
    
    # Вибираємо лише точки з x < 0.75 для уникнення розриву
    x_points = x_points_all[x_points_all < 0.75]
    y_points = y_points_all[:len(x_points)]
    
    print(f"Використано {len(x_points)} експериментальних точок із {len(x_points_all)}")
    
    # Створюємо інтервали з похибкою 20%
    error_percentage = 20
    y_lower = [y * (1 - error_percentage/100) for y in y_points]
    y_upper = [y * (1 + error_percentage/100) for y in y_points]
    y_intervals = list(zip(y_lower, y_upper))
    
    # Параметри алгоритму
    initial_params = [0.5, 5.0]  # Початкові значення [a0, b0]
    radius = 1.2          # Початковий радіус пошуку
    k_dec = 0.8           # Коефіцієнт зменшення радіуса
    max_iterations = 200  # Максимальна кількість ітерацій
    max_no_improvement = 10  # Кількість ітерацій без покращення
    num_vectors = 100     # Кількість випадкових векторів
    
    # Запускаємо алгоритм оптимізації
    best_params, best_objective, iterations, execution_time, history = random_search(
        x_points, y_intervals, initial_params, radius, k_dec, max_iterations, 
        max_no_improvement, num_vectors
    )
    
    # Виводимо результати
    print("\nРезультати оптимізації:")
    print(f"Оптимальні значення: a = {best_params[0]:.6f}, b = {best_params[1]:.6f}")
    print(f"Значення функції мети = {best_objective:.6f}")
    print(f"Кількість ітерацій = {iterations}")
    print(f"Час виконання = {execution_time:.2f} секунд")
    
    # Перевіряємо значення функції в важливих точках
    try:
        value_at_zero = system_function(0.0, best_params[0], best_params[1])
        print(f"Значення функції при x=0: {value_at_zero:.6f}")
    except Exception:
        print("Неможливо обчислити значення функції при x=0")
    
    # Візуалізуємо результати
    visualize_results(x_points, y_points, y_lower, y_upper, best_params, 
                      x_points_all, y_points_all, history)

# Запускаємо програму
if __name__ == "__main__":
    main()