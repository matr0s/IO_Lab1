import json

class IntervalNumber:
    """Клас для роботи з інтервальними числами [a, b]"""
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    
    def __str__(self):
        return f"[{self.lower}, {self.upper}]"
    
    def __sub__(self, other):
        # Віднімання інтервалів: [a, b] - [c, d] = [a - d, b - c]
        return IntervalNumber(self.lower - other.upper, self.upper - other.lower)
    
    def __mul__(self, other):
        # Множення інтервалів: [a, b] * [c, d] = [min(ac, ad, bc, bd), max(ac, ad, bc, bd)]
        products = [
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper
        ]
        return IntervalNumber(min(products), max(products))

class IntervalMatrix:
    """Клас для роботи з матрицями інтервальних чисел"""
    def __init__(self, matrix=None):
        self.matrix = matrix if matrix else []
    
    @classmethod
    def from_list(cls, data):
        """Створення матриці з вкладених списків"""
        matrix = []
        for row in data:
            matrix_row = []
            for cell in row:
                matrix_row.append(IntervalNumber(cell[0], cell[1]))
            matrix.append(matrix_row)
        return cls(matrix)
    
    def get_dimensions(self):
        """Повертає розміри матриці (рядки, стовпці)"""
        if not self.matrix:
            return 0, 0
        return len(self.matrix), len(self.matrix[0])
    
    def __mul__(self, other):
        """Множення матриць"""
        rows_a, cols_a = self.get_dimensions()
        rows_b, cols_b = other.get_dimensions()
        
        # Перевірка розмірностей
        if cols_a != rows_b:
            raise ValueError("Розміри матриць несумісні для множення")
        
        # Створення результуючої матриці
        result_matrix = []
        for i in range(rows_a):
            result_row = []
            for j in range(cols_b):
                # Сума добутків елементів рядка A і стовпця B
                sum_interval = IntervalNumber(0, 0)
                for k in range(cols_a):
                    product = self.matrix[i][k] * other.matrix[k][j]
                    # Додавання інтервалів шляхом прямого обчислення
                    sum_interval = IntervalNumber(
                        sum_interval.lower + product.lower,
                        sum_interval.upper + product.upper
                    )
                result_row.append(sum_interval)
            result_matrix.append(result_row)
        
        return IntervalMatrix(result_matrix)
    
    def __sub__(self, other):
        """Віднімання матриць"""
        rows_a, cols_a = self.get_dimensions()
        rows_b, cols_b = other.get_dimensions()
        
        # Перевірка розмірностей
        if rows_a != rows_b or cols_a != cols_b:
            raise ValueError("Розміри матриць несумісні для віднімання")
        
        # Створення результуючої матриці
        result_matrix = []
        for i in range(rows_a):
            result_row = []
            for j in range(cols_a):
                result_row.append(self.matrix[i][j] - other.matrix[i][j])
            result_matrix.append(result_row)
        
        return IntervalMatrix(result_matrix)
    
    def __str__(self):
        """Зручне виведення матриці"""
        result = ""
        for row in self.matrix:
            row_str = [str(cell) for cell in row]
            result += "  ".join(row_str) + "\n"
        return result

def main():
    # Зчитування матриць з конфігураційного файлу
    try:
        with open("interval_matrices.json", 'r') as file:
            data = json.load(file)
            
        # Створення матриць
        A = IntervalMatrix.from_list(data['matrix_a'])
        B = IntervalMatrix.from_list(data['matrix_b'])
        C = IntervalMatrix.from_list(data['matrix_c'])
        
        # Виведення вхідних матриць
        print("\n" + "="*60)
        print("Інтервальний калькулятор матриць")
        print("="*60)
        
        print("\nМатриця A:")
        print(A)
        
        print("\nМатриця B:")
        print(B)
        
        print("\nМатриця C:")
        print(C)
        
        # Обчислення Q = B*C - A
        print("\nФормула: Q = B × C - A")
        
        # Спочатку обчислюємо B*C
        BC = B * C
        
        # Потім обчислюємо Q = BC - A
        Q = BC - A
        
        print("\nРезультуюча матриця Q:")
        print(Q)
        
        print("="*60)
        
    except Exception as e:
        print(f"Помилка: {e}")

if __name__ == "__main__":
    main()