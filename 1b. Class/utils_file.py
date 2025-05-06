class utils():
    
    
    """Create a class and function, and 
    list out the items in the list."""
    
    def Subfields():
        sub_list = ["Machine Learning", "Neural Networks", "Vision." "Robotics", "Speech Processing", "Natural Language Processing"]
        print(f"Sub-fields in AI are:")
        for item in sub_list:
            print(item)
          
          
            
        """Create a function that checks whether 
        the given number is Odd or Even."""       
            
    def OddEven():
        number = int(input(f"Enter a number: ")) #Also can use this method - def OddEven(number): print(f"Enter a number: {number}")
        if number % 2 == 0:
            print(f"{number} is Even number.")
        else:
            print(f"{number} is Odd number.")
        #return number
        
        
        
        """Create a function that tells elegibility of marriage for male and 
        female according to their age limit like 21 for male and 18 for female."""
        
    def Elegible():
        gender = input("Your Gender:") #Mention Male or Female
        age = int(input("Your Age:"))
        if (gender.upper() == "MALE" and age >= 21) or (gender.upper() == "FEMALE" and age >= 18):
            print("ELIGIBLE")
        else:
            print("NOT ELIGIBLE")
            
            
            
        """"calculate the percentage of your 10th marks."""        
            
    def percentage():
        Subject1 = 98
        print(f"Subject1= {Subject1}")
        Subject2 = 87 
        print(f"Subject2= {Subject2}")
        Subject3 = 95 
        print(f"Subject3= {Subject3}")
        Subject4 = 95 
        print(f"Subject4= {Subject4}")
        Subject5 = 93 
        print(f"Subject5= {Subject5}")
        Total = Subject1 + Subject2 + Subject3 + Subject4 + Subject5
        Percentage = (Total / 500) * 100
        print(f"Total :  {Total} \nPercentage :  {Percentage}")
        
        
        
        """"print area and perimeter of triangle using class and functions."""
        
    def triangle(): 
        Height = 32 
        print(f"Height:{Height}")
        Breadth = 34 
        print(f"Breadth:{Breadth}")
        Area_formula = (Height*Breadth)/2 
        print(f"Area formula: (Height*Breadth)/2")
        print(f"Area of Triangle:  {Area_formula}")
        Height1 = 2 
        print(f"Height1:{Height1}")
        Height2 = 4 
        print(f"Height2:{Height2}")
        Breadth = 4 
        print(f"Breadth:{Breadth}")
        Perimeter_formula = Height1+Height2+Breadth 
        print(f"Perimeter formula: Height1+Height2+Breadth")
        print(f"Perimeter of Triangle:  {Perimeter_formula}")