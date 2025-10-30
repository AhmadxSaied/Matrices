#include <stdio.h>
#include <math.h>
#include <malloc.h>
long Binary_to_Decimal(char* Binary){
    char* Temp = Binary; 
    long Decimal_Value = 0;
    long max_Power=-1;
    while(*Temp !='\0'){
        Temp++;
        max_Power++;
    }
    Temp=Binary;
    while (*Temp!='\0')
    {
        char Bit= *Temp;
        if(Bit == '1'){
            Decimal_Value += pow(2,max_Power); 
        }else if(Bit=='0'){
            Decimal_Value+=0;
        }else{
            fprintf(stderr, "Wrong Binary Number...\n");
            return -1;
        }
        max_Power--;
        Temp++;
    }
    return Decimal_Value;
}
void Decimal_to_Binary(){
    printf("Enter The Decimal Number: ");
    long Number =0;
    scanf("%ld",&Number);
    long Temp=Number;
    char* Binary_String=malloc(sizeof(char)*50);
    Binary_String[0]='\0';
    int index=1;
    while(Temp!=0){
        if(Temp%2 ==1){
            Binary_String[index]='1';
        }else{
            Binary_String[index]='0';
        }
        index++;
        Temp/=2;
    }
    char* Binary_point= &Binary_String[--index];
    while(*Binary_point!='\0'){
        printf("%c",*Binary_point);
        Binary_point--;
    }
}
int main()
{
    char Binary[50]= { 0 };
    printf("Enter the binary Number: ");
    scanf("%s",Binary);
    long l=Binary_to_Decimal(Binary);
    printf("%ld",l);
    // printf("Decimal :%ld",Binary);
    return 0;
}
