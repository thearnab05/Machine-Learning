import pandas as pd

def main():
    print("Hello, Machine Learning World!")
    print("Testing pandas installation by creating a simple DataFrame:")
    
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Paris', 'London']
    }
    
    df = pd.DataFrame(data)
    print("\nOur first DataFrame:")
    print(df)
    
if __name__ == "__main__":
    main()
