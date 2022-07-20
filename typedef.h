typedef struct{
    int x,y;
} Mv;

typedef struct{
    int nCPs;
    Mv LT, RT, LB;
} Cpmvs;

enum QPs{
    QP22 = 0,
    QP27 = 1,
    QP32 = 2,
    QP37 = 3
};