#Function to add two numbers

def add(x,y):
    return x+y

#Function to substarct two numbers

def sub(x,y):
    return x-y

#Function to multiply two numbers

def multiply(x,y):
    return x*y

#Function to divide two numbers

def division(x,y):
    if(y==0):
        return "Error! Division by zero not allowed"
    else:   
        return x/y
    
def calculator():
    print("Select operation:")   
    print("1. Add") 
    print("2. Subtract")  
    print("3. Multiply")   
    print("4. Divide") 
    
    while True:
        # Take input form the user
        choice = input("Enter choices(1/2/3/4) :") 
        
        # Check if the input is one of the four options
        
        if choice in ['1','2','3','4'] :
            num1=float(input("Enter first number: "))
            num2=float(input("Enter second number: "))
            
            if choice == '1':
                print(f"{num1} + {num2} = {add(num1,num2)}")
                
            
            if choice == '2':
                print(f"{num1}-{num2} = {sub(num1,num2)}")
                
            if choice =='3':
                print(f"{num1} * {num2} = {multiply(num1,num2)}") 
                
            if choice == '4':
                print(f"{num1} | {num2} = {division(num1,num2)}")
                
            
        # Option to exit the loop
        
        next_calculation = input("Do you want to perform another calculation(yes/no) : ")
        if next_calculation.lower() != 'yes' :
            break
        print("Exiting Calculator,Goodbye!")


# Call the calculator
calculator()        