class Interval:
    """
    Клас для роботи з інтервалами.
    Кожен інтервал представлений двома межами: лівою та правою.
    """
    def __init__(self, left, right):
        # Конвертуємо вхідні значення у float для більшої точності 
        self.left = float(left)
        self.right = float(right)
    
    def __str__(self):
        # Виводимо значення як цілі числа, щоб краще читалось
        return f"[{self.left:.0f}, {self.right:.0f}]"
    
    def __add__(self, other):
        # Додавання інтервалів виконується покомпонентно:
        # ліва межа + ліва межа, права межа + права межа
        return Interval(self.left + other.left, self.right + other.right)
    
    def __sub__(self, other):
        # При відніманні інтервалів ліва межа результату - це різниця
        # лівої межі першого і правої межі другого інтервалу
        return Interval(self.left - other.right, self.right - other.left)
    
    def __mul__(self, other):
        # При множенні інтервалів потрібно розглянути всі можливі
        # комбінації множення меж і вибрати мінімум та максимум
        products = [
            self.left * other.left,
            self.left * other.right,
            self.right * other.left,
            self.right * other.right
        ]
        return Interval(min(products), max(products))
    
    def __truediv__(self, other):
        # Перевіряємо чи містить дільник нуль
        if other.left <= 0 <= other.right:
            # Створюємо список для зберігання всіх можливих результатів ділення
            quotients = []
            
            # Збираємо всі допустимі значення для ділення (крім нуля)
            divisors = []
            if other.left < 0:
                divisors.append(other.left)
            if other.right > 0:
                divisors.append(other.right)
            
            # Для кожного допустимого дільника обчислюємо результати
            for divisor in divisors:
                quotients.extend([
                    self.left / divisor,   # Ділимо ліву межу
                    self.right / divisor   # Ділимо праву межу
                ])
            
            # Перевіряємо чи маємо хоч якісь результати
            if not quotients:
                raise ValueError("Ділення на інтервал, що містить тільки нуль, неможливе")
            
            # Створюємо новий інтервал з мінімального та максимального результатів
            return Interval(min(quotients), max(quotients))
        
        # Якщо дільник не містить нуль, виконуємо звичайне ділення інтервалів
        quotients = [
            self.left / other.left,    # Ліва межа на ліву
            self.left / other.right,   # Ліва межа на праву
            self.right / other.left,   # Права межа на ліву
            self.right / other.right   # Права межа на праву
        ]
        
        # Повертаємо новий інтервал з мінімального та максимального результатів
        return Interval(min(quotients), max(quotients))


class ComplexInterval:
    """
    Клас для роботи з комплексними інтервальними числами.
    Кожне число це два інтервали: для дійсної та уявної частини.
    """
    def __init__(self, real, imag):
        self.real = real  # Інтервал для дійсної частини
        self.imag = imag  # Інтервал для уявної частини
    
    def __str__(self):
        return f"{self.real} + {self.imag}i"
    
    def __add__(self, other):
        # Додавання комплексних інтервалів виконується покомпонентно
        return ComplexInterval(
            self.real + other.real,  # Додаємо дійсні частини
            self.imag + other.imag   # Додаємо уявні частини
        )
    
    def __sub__(self, other):
        # Віднімання також виконується покомпонентно
        return ComplexInterval(
            self.real - other.real,
            self.imag - other.imag
        )
    
    def __mul__(self, other):
        # Множення комплексних інтервалів за формулою (a+bi)(c+di)=(ac-bd)+(ad+bc)i
        real_part = self.real * other.real - self.imag * other.imag
        imag_part = self.real * other.imag + self.imag * other.real
        return ComplexInterval(real_part, imag_part)

def calculate_first_expression():
    """Обчислення першого виразу: [-1,6]·[-1,6] + ([-3,3]-[1,5])/[-2,7] - [2,6]"""
    print("\nОбчислення першого виразу:")
    
    # 1. Обчислюємо [-1,6]·[-1,6]
    int1 = Interval(-1, 6)
    mul_result = int1 * int1
    print(f"1) [-1,6]·[-1,6] = {mul_result}")
    
    # 2. Обчислюємо [-3,3]-[1,5]
    int2 = Interval(-3, 3)
    int3 = Interval(1, 5)
    sub_result = int2 - int3
    print(f"2) [-3,3]-[1,5] = {sub_result}")
    
    # 3. Ділимо на [-2,7]
    int4 = Interval(-2, 7)
    div_result = sub_result / int4
    print(f"3) ({sub_result})/[-2,7] = {div_result}")
    
    # 4. Додаємо результати та віднімаємо [2,6]
    int5 = Interval(2, 6)
    final_result = mul_result + div_result - int5
    print(f"4) {mul_result} + {div_result} - [2,6] = {final_result}")
    
    return final_result

def calculate_second_expression():
    """Обчислення другого виразу: ([3,8]-[0,4])/([1,1]-[-3,3]) - [0,3]·[-2,2]"""
    print("\nОбчислення другого виразу:")
    
    # 1. Обчислюємо [3,8]-[0,4]
    int1 = Interval(3, 8)
    int2 = Interval(0, 4)
    sub_result1 = int1 - int2
    print(f"1) [3,8]-[0,4] = {sub_result1}")
    
    # 2. Обчислюємо [1,1]-[-3,3]
    int3 = Interval(1, 1)
    int4 = Interval(-3, 3)
    sub_result2 = int3 - int4
    print(f"2) [1,1]-[-3,3] = {sub_result2}")
    
    # 3. Ділимо перший результат на другий
    div_result = sub_result1 / sub_result2
    print(f"3) {sub_result1}/{sub_result2} = {div_result}")
    
    # 4. Обчислюємо [0,3]·[-2,2]
    int5 = Interval(0, 3)
    int6 = Interval(-2, 2)
    mul_result = int5 * int6
    print(f"4) [0,3]·[-2,2] = {mul_result}")
    
    # 5. Віднімаємо результати
    final_result = div_result - mul_result
    print(f"5) {div_result} - {mul_result} = {final_result}")
    
    return final_result

def main():
    print("Розв'язання варіанту 16")
    print("Додавання до основ інтервалу [-2,2]")
    
    # Обчислення основ комплексних чисел
    base1 = calculate_first_expression()
    base2 = calculate_second_expression()
    
    # Створення уявної частини
    imaginary = Interval(-2, 2)
    print(f"\nІнтервал уявної частини: {imaginary}")
    
    # Створення комплексних інтервалів
    z1 = ComplexInterval(base1, imaginary)
    z2 = ComplexInterval(base2, imaginary)
    
    print("\nОтримані комплексні інтервали:")
    print(f"A = {z1}")
    print(f"B = {z2}")
    
    # Виконання операцій з комплексними інтервалами
    print("\nРезультати операцій:")
    addition = z1 + z2
    subtraction = z1 - z2
    multiplication = z1 * z2
    
    print(f"A + B = {addition}")
    print(f"A - B = {subtraction}")
    print(f"A × B = {multiplication}")

if __name__ == "__main__":
    main()