CWRKSET
      SUBROUTINE WRKSET(A,MDA,T,P,N,G,B,TAU,MDF,SCALE,ITNO,
     1 DIAG,ACTIVE,BND,INACT,Q,H,GNDNRM,P4,C,MDC,M,F,MDG,
     2 X,HFUNC,FFUNC,FUNCEV,JACEV,
     3 P2,P3,DX,V1,D2,D3,RANKC2,D1NRM,DNRM,B1NRM,D,GMAT,
     4 P1,V,D1,FMAT,RANKA,GRES,NOHOUS,TIME,DEL,PIVOT,V2,S,U)
      INTEGER MDA,T,P,N,MDF,BND,SCALE,Q,ITNO,RANKA,NOHOUS,TIME,
     1 MDC,M,MDG,FUNCEV,JACEV,RANKC2
      INTEGER ACTIVE(1),INACT(1),P1(1),P2(1),P3(1),P4(1)
      DOUBLE PRECISION
     *     TAU,GRES,GNDNRM,D1NRM,DNRM,B1NRM
      DOUBLE PRECISION
     *     A(MDA,N),G(N),B(1),DIAG(1),PIVOT(1),V(1),D1(1),FMAT(MDF,N),
     1 V1(M),V2(N),S(N),U(N),H(1),C(MDC,N),F(M),X(N),DX(N),D2(1),
     2 D3(N),D(M),GMAT(MDG,N)
      LOGICAL DEL
      EXTERNAL HFUNC,FFUNC
C
C     DETERMINE THE CURRENT WORKING SET BY ESTIMATING THE LAGRANGE
C     MULTIPLIERS AND COMPUTE THE GN-SEARCH DIRECTION
C     IN ORDER TO DO THAT THE OVER DETERMINED SYSTEM
C                T
C               A *V = G     IS SOLVED FOR V
C
C     THEN V HOLDS A FIRST ORDER ESTIMATE OF THE MULTIPLIERS
C     THE MATRIX A IS DECOMPOSED AS              T
C                                      (L@D0) = P1 *A*Q1*FMAT
C     WHERE L IS T*T LOWER TRIANGULAR
C           P1 IS T*T PERMUTATION MATRIX
C           Q1 IS N*N ORTHOGONAL MATRIX
C           FMAT IS N*N ORTHOGONAL OR THE IDENTITY
C     THE LAGRANGE ESTIMATES CORRESPONDING TO INEQUALITIES ARE
C     INSPECTED BY THEIR SIGNS AND COMPARED RELATIVELY TO THE
C     RESIDUAL  A(TR)*V-G.
C     IF NO ESTIMATE V(I) IMPLIES DELETION ANOTHER ESTIMATE (THE GN-
C     ESTIMATE) IS COMPUTED.
C     THE GN-ESTIMATE IS THE SOLUTION (U) OF THE OVERDETERMINED SYSTEM
C          T        T         T
C         A *U = G+C *C*DX = C *(F+C*DX)
C
C     WHERE DX IS THE GN-SEARCH DIRECTION.
C
C     ON ENTRY@D
C
C     A(,)    REAL DOUBLY SUBSCRIPTED ARRAY OF DIMENSION MDA*N
C             CONTAINIG THE MATRIX A IN THE UPPER LEFT T*N RECTANGLE
C             (IF SCALING HAS BEEN USED ARRAY A CONTAINS DIAG*A )
C     MDA     INTEGER SCALAR CONTAINING LEADING DIMENSION OF ARRAY A
C     T       INTEGER SCALAR CONTAINING NUMBER OF CONSTRAINTS IN
C             CURRENT WORKING SET
C     P       INTEGER SCALAR CONTAINING NUMBER OF EQUALITY CONSTRAINTS
C     N       INTEGER SCALAR CONTAINING NUMBER OF PARAMETERS
C     G()     REAL SINGLY SUBSCRIPTED ARRAY OF DIMENSION N
C             CONTAINING THE GRADIENT OF THE OBJECTIVE FUNCTION
C     B()     REAL SINGLY SUBSCRIPTED ARRAY OF DIMENSION T
C             CONTAINING -H (OR -DIAG*H IF SCALING HAS BEEN DONE)
C             WHERE H(I) I=1,2,....,T ARE THE VALUES OF THE CONSTRAINTS
C             IN CURRENT WORKING SET
C     TAU     REAL SCALAR CONTAINING A SMALL VALUE USED TO DETERMINE
C             PSEUDO RANK OF MATRIX A
C     MDF     INTEGER SCALAR CONTAINING LEADING DIMENSION OF ARRAY FMAT
C     ADD     LOGICAL SCALAR = TRUE IF SOME CONSTRAINTS WERE ADDED
C             IN THE LATEST STEP
C                               = FALSE IF NOT SO
C     SCALE   INTEGER SCALAR =0 IF NO SCALING HAS BEEN DONE
C                   > 0 IF ROW SCALING OF MATRIX A HAS BEEN DONE
C     ITNO    INTEGER SCALAR CONTAINING THE ITERATION NUMBER
C     DIAG()  REAL SINGLY SUBSCRIPTED ARRAY OF DIMENSION T
C             CONTAINING THE DIAGONAL ELEMENTS IN THE SCALING MATRIX
C             DIAG (IF SCALING HAS NOT BEEN DONE DIAG(I)= THE LENGTH
C                   OF ROW NO I IN THE ORIGINAL MATRIX A)
C     ACTIVE()INTEGER SINGLY SUBSCRIPTED ARRAY OF DIMENSION T
C             CONTAINING INDECES FOR THE CONSTRAINTS IN CURRENT
C             WORKING SET
C     BND     INTEGER SCALAR CONTAINING MIN(L,N)
C     INACT() INTEGER SINGLY SUBSCRIPTED ARRAY OF DIMENSION Q
C             CONTAINING INDECES FOR CONSTRAINTS NOT IN CURRENT
C             WORKING SET
C     Q       INTEGER SCALAR CONTAINING NUMBER OF CONSTRAINTS NOT IN
C             CURRENT WORKING SET
C     H()     REAL SINGLY SUBSCRIPTED ARRAY OF DIMENSION L
C             CONTAINING THE VALUE OF THE CONSTRAINTS AT CURRENT POINT
C     GNDNRM  REAL SCALAR CONTAINING II D II AT POINT X(K-1) COMPUTED
C             BY THE GAUSS-NEWTON METHOD
C     P4()    INTEGER SINGLY SUBSCRIPTED ARRAY OF DIMENSION L
C             CONTAINING INFO. TO COMPUTE A*DX FOR INACTIVE CONSTRAINTS
C     C(,)    REAL DOUBLY SUBSCRIPTED ARRAY OF DIMENSION MDC*N
C             CONTAINING THE JACOBIAN OF THE RESIDUALS
C     MDC     INTEGER SCALAR CONTAINING LEADING DIMENSION OF ARRAY C
C     M       INTEGER SCALAR CONTAINING NUMBER OF RESIDUALS
C     F()     REAL SINGLY SUBSCRIPTED ARRAY OF DIMENSION M
C             CONTAINING THE VALUE OF THE RESIDUALS
C     MDG     INTEGER SCALAR CONTAINING LEADING DIMENSION OF ARRAY GMAT
C     X()     REAL SINGLY SUBSCRIPTED ARRAY OF DIMENSION N
C             CONTAINING THE CURRENT POINT
C     HFUNC   SUBROUTINE NAMES. SEE EXPLANATION IN SUBROUTINE NLSNIP
C     FFUNC
C     FUNCEV  INTEGER SCALAR CONTAINING @ OF TIMES THE RESIDUALS
C             ARE EVALUATED
C     JACEV   INTEGER SCALAR CONTAINING @ OF TIMES THE JACOBIANS
C             ARE EVALUATED
C
C     ON RETURN@D
C                                        T
C     A(,)    CONTAINS MATRIX L  FROM  P1 *A*Q1*FMAT = (L@D0)
C             AS THE T FIRST COLUMNS AND INFO. TO FORM MATRIX Q1
C     T       CONTAINS NUMBER OF CONSTRAINTS IN UPDATED WORKING SET
C     B()     CONTAINS  T
C                     P1 *B
C     ACTIVE()CONTAINS INDECES FOR THE CONSTRAINTS IN THE
C             UPDATED WORKING SET
C     INACT() HOLDS INDECES FOR THE CONSTRAINTS NOT IN THE
C             UPDATED WORKING SET
C     Q       CONTAINS NUMBER OF CONSTRAINTS IN INACTIVE SET
C     P1()    INTEGER SINGLY SUBSCRIPTED ARRAY OF DIMENSION T
C             REPRESENTS THE PERMUTATION MATRIX P1 IN
C             P1(TR)*A*Q1*FMAT
C     P2()    INTEGER SINGLY SUBSCRIPTED ARRAY OF DIMENSION RANKA
C             REPRESENTING PERMUTATION MATRIX P2 (IF IT IS USED)
C     P3()    INTEGER SINGLY SUBSCRIPTED ARRAY OF DIMENSION N-RANKA
C             REPRESENTING PERMUTATION MATRIX P3
C     DX()    REAL SINGLY SUBSCRIPTED ARRAY OF DIMENSION N
C             CONTAINS THE GAUSS-NEWTON SEARCH DIRECTION
C     V1()    REAL SINGLY SUBSCRIPTED ARRAY OF DIMENSION M+T
C             CONTAINS THE COMPOUND VECTOR  (C*DX)
C                                           (A*DX)
C     D2()    REAL SINGLY SUBSCRIPTED ARRAY OF DIMENSION RANKA
C             CONTAINS INFO. TO FORM Q2 (IF IT IS USED)
C     D3()    REAL SINGLY SUBSCRIPTED ARRAY OF DIMENSION N-RANKA
C             CONTAINS INFO. TO FORM Q3
C     RANKC2  INTEGER SCALAR-CONTAINS PSEUDO RANK OF MATRIX C2
C                                      -1                     T
C             DENOTE  (D)=(D1)= -F-C1*L  *B1    WHERE  (B1)=Q2 *B
C                         (D2)                         (B2)
C             D1 IS RANKC2*1            B1 IS RANKA*1
C             THEN
C     D1NRM   REAL SCALAR-CONTAINS EUCLIDEAN NORM OF D1
C     DNRM    REAL SCALAR-CONTAINS EUCLIDEAB NORM OF D
C     B1NRM   REAL SCALAR-CONTAINS EUCLIDEAN NORM OF B1
C     D()     REAL SINGLY SUBSCRIPTED ARRAY OF DIMENSION M
C             CONTAINING    T
C                         Q3 *(-F-C1*DY1)
C     GMAT(,) REAL DOUBLY SUBSCRIPTED ARRAY OF DIMENSION MDG*N
C             CONTAINS MATRIX R AND INFO. TO FORM Q2 (IF IT IS USED)
C
C     P4()    CONTAINS POSSIBLE CHANGED INFO. TO COMPUTE A*DX FOR
C             INACTIVE CONSTRAINTS
C     V()     REAL SINGLY SUBSCRIPTED ARRAY OF DIMENSION T
C             CONTAINS THE GN-MULTIPLIER ESTIMATES WHEN T=RANKA.
C             OTHERWISE V EQUALS THE USUAL 1ST-ORDER ESTIMATES
C     D1()    REAL SINGLY SUBSCRIPTED ARRAY OF DIMENSION T
C             CONTAINS INFO. TO FORM Q1
C     FMAT(,) REAL DOUBLY SUBSCRIPTED ARRAY OF DIMENSION MDF*N
C             CONTAINS A N*N MATRIX (A PRODUCT OF GIVENS ROTATION
C             MATRICES) IF TIME>2
C     RANKA   INTEGER SCALAR-CONTAINS THE PSEUDO RANK OF MATRIX A
C     GRES    REAL SCALAR-CONTAINS THE EUCLIDEAN NORM OF THE RESIDUAL
C               T           T      T
C              A *V-G  OR  A *V-G-C *C*DX     DEPENDING ON WHETHER
C             V IS A GN- OR ORDINARY 1ST ORDER ESTIMATE
C     NOHOUS  INTEGER SCALAR NUMBER OF HOUSEHOLDER TRANSFORMATIONS
C             DONE TO TRANSFORM ORIGINAL A TO LOWER TRIANGULAR FORM
C     TIME    INTEGER SCALAR-CONTAINS NUMBER OF DELETIONS+2
C     DEL     LOGICAL SCALAR=TRUE IF ANY CONSTRAINT IS DELETED
C             =FALSE IF NO DELETION IS DONE
C     PIVOT() REAL SINGLY SUBSCRIPTED ARRAY OF DIMENSION T
C             CONTAINS INFO. TO FORM Q1
C
C     WORKING AREAS@D
C
C     V2()       REAL SINGLY SUBSCRIPTED ARRAYS ALL OF
C     S(),U()    DIMENSION N
C
C     COMMON VARIABLES CONTAINING INFORMATION CONCERNING PREVIOUS
C     TWO POINTS. THE SUFFICES KM2 AND KM1 IN THE NAMES OF THE
C     VARIABLES REPRESENT TIME STEP K-2 AND K-1
C     THESE VARIABLES ARE UPDATED ONLY INSIDE THE ROUTINE EVREST
C
      INTEGER RKAKM2,RKCKM2,KODKM2,RKAKM1,RKCKM1,KODKM1,TKM2,TKM1
      DOUBLE PRECISION
     *     BETKM2,D1KM2,DKM2,FSQKM2,HSQKM2,B1KM2,DXNKM2,ALFKM2,
     1 BETKM1,D1KM1,DKM1,FSQKM1,HSQKM1,B1KM1,DXNKM1,ALFKM1,
     2 PGRESS,PRELIN
      COMMON /PREC/ BETKM2,D1KM2,DKM2,FSQKM2,HSQKM2,B1KM2,DXNKM2,
     1 ALFKM2,RKAKM2,RKCKM2,TKM2,KODKM2,
     2 BETKM1,D1KM1,DKM1,FSQKM1,HSQKM1,B1KM1,DXNKM1,ALFKM1,
     3 RKAKM1,RKCKM1,TKM1,KODKM1,
     4 PGRESS,PRELIN
C
C     COMMON VARIABLES CONTAINING INFORMATION OF RESTART STEPS
C
      DOUBLE PRECISION
     *    BESTRK,BESTPG
      INTEGER NRREST,LATTRY
      COMMON /BACK/ BESTRK,BESTPG, NRREST,LATTRY
C
C     COMMON VARIABLES CONTAINING MACHINE DEPENDENT CONSTANTS
C     DRELPR = DOUBLE RELATIVE PRECISION
C
      DOUBLE PRECISION
     *    DRELPR
      COMMON /MACHIN/ DRELPR
C
C     INTERNAL VARIABLES
C
      INTEGER I,IER,I2,I3,NOEQ,J,L
      DOUBLE PRECISION
     *     TOL,RES
*      write(10,*) 'In WRKSET: t= ',t
      J=0
      L=Q+T
      TIME=1
      TOL=DSQRT(DBLE(T))*TAU
C
C     COMPUTE FIRST ORDER ESTIMATES OF LAGRANGE MULTIPLIERS
C
      DEL=.FALSE.
      CALL MULEST(TIME,A,MDA,T,N,G,B,J,TOL,D1,FMAT,MDF,PIVOT,
     1 P1,SCALE,DIAG,V,RANKA,GRES,S,U,V2)
      NOHOUS=RANKA
C
C     DETERMINE WHICH (IF ANY) CONSTRAINT THAT SHOULD BE DELETED
C
      IF((M+T) .LE. N) GOTO 80
      CALL SIGNCH(TIME,P1,V,T,ACTIVE,BND,D1KM1,GNDNRM,ITNO,
     1 SCALE,DIAG,GRES,H,P,L,J,NOEQ,U,V2)
      IF(NOEQ.EQ.0) GOTO 100
*      write(10,*) 'Index for delete constraint (NOEQ=) ',noeq
C
C     UPDATE DECOMPOSED MATRIX A BY DELETING APPROPRIATE ROW
C     FROM MATRIX L
C
      DEL=.TRUE.
      CALL REORD(A,MDA,T,N,B,J,NOEQ,ACTIVE,INACT,Q,P4,U,SCALE,DIAG)
C
C     UPDATE THE THE LOWER TRIANGULAR MATRIX L
C
      CALL MULEST(TIME,A,MDA,T,N,G,B,J,TOL,D1,FMAT,MDF,PIVOT,P1,
     1 SCALE,DIAG,V,RANKA,GRES,S,U,V2)
*      write(10,*) 'RANKA after call mulest ',ranka
C
C     COMPUTE THE GN-SEARCH DIRECTION
C
      CALL GNSRCH(TIME,A,MDA,T,N,D1,P1,RANKA,NOHOUS,B,FMAT,MDF,C,MDC,
     1 M,F,PIVOT,TAU,MDG,SCALE,DIAG,INACT,Q,P4,P2,P3,DX,V1,D2,D3,
     2 RANKC2,D1NRM,DNRM,B1NRM,D,S,U,GMAT)
C
C     TEST FOR FEASIBLE DIRECTION
C
      I2=INACT(Q)
      I3=M+L
      IF((V1(I3).GE.-H(I2)).AND.(V1(I3).GT. 0.0D0)) GOTO 200
      DEL=.FALSE.
C
C     NOT FEASIBLE.
C     RECOMPUTE JACOBIANS, REARANGE WORKING- AND INACTIVE SETS
C
*      write(10,*) 'Not feasible direction: rearange'
      CALL NEWPNT(X,N,H,L,F,M,HFUNC,FFUNC,MDA,MDC,FUNCEV,A,C,B,D,IER)
      JACEV=JACEV+1
      J = Q
      CALL ADDIT(ACTIVE,INACT,T,Q,J)
      CALL EQUAL(B,L,A,MDA,N,ACTIVE,T,P,P4)
      CALL GRAD(C,MDC,M,N,F,G)
      CALL EVSCAL(SCALE,A,MDA,T,N,B,DIAG)
      CALL UNSCR(ACTIVE,BND,L,P)
      J=0
      TIME=1
      CALL MULEST(TIME,A,MDA,T,N,G,B,J,TOL,D1,FMAT,MDF,PIVOT,P1,
     1 SCALE,DIAG,V,RANKA,GRES,S,U,V2)
      NOHOUS=RANKA
   80 CONTINUE
      CALL GNSRCH(TIME,A,MDA,T,N,D1,P1,RANKA,NOHOUS,B,FMAT,MDF,C,MDC,
     1 M,F,PIVOT,TAU,MDG,SCALE,DIAG,INACT,Q,P4,P2,P3,DX,V1,D2,D3,
     2 RANKC2,D1NRM,DNRM,B1NRM,D,S,U,GMAT)
      GOTO 110
  100 CONTINUE
C
C     NO FIRST ORDER ESTIMATE IMPLIES DELETION OF A CONSTRAINT.
C     COMPUTE GN-ESTIMATE MULTIPLIERS
C
      CALL GNSRCH(TIME,A,MDA,T,N,D1,P1,RANKA,NOHOUS,B,FMAT,MDF,C,MDC,
     1 M,F,PIVOT,TAU,MDG,SCALE,DIAG,INACT,Q,P4,P2,P3,DX,V1,D2,D3,
     2 RANKC2,D1NRM,DNRM,B1NRM,D,S,U,GMAT)
C
C     COMPUTE GN-ESTIMATES V(I), I=1,2,....,T
C
  110 continue
      if(kodkm1.ne.1) goto 200
      IF((T.NE.RANKA).OR.(RANKC2.NE.MIN0(M,N-RANKA))) GOTO 200
      CALL LEAEST(A,MDA,T,F,M,V1,C,MDC,P1,SCALE,DIAG,S,V,RES)
C
C     DETERMINE WHICH CONSTRAINT SHOULD BE DELETED
C
      CALL SIGNCH(TIME,P1,V,T,ACTIVE,BND,D1NRM,DNRM,ITNO,SCALE,
     1 DIAG,RES,H,P,L,J,NOEQ,S,V2)
      IF(NOEQ .EQ. 0) GOTO 200
      DEL=.TRUE.
C
C     DELETE APPROPRIATE ROW IN MATRIX A AND UPDATE MATRIX L
C
      CALL REORD(A,MDA,T,N,B,J,NOEQ,ACTIVE,INACT,Q,P4,S,SCALE,DIAG)
      CALL MULEST(TIME,A,MDA,T,N,G,B,J,TOL,D1,FMAT,MDF,PIVOT,P1,
     1 SCALE,DIAG,V,RANKA,GRES,S,D,V2)
      IER=2
      CALL FFUNC(X,N,F,M,IER,C,MDC)
      IF(IER.NE.0) GOTO 150
      CALL JACDIF(X,N,F,M,FFUNC,C,MDC,D,IER)
      FUNCEV=FUNCEV+N
  150 CONTINUE
      JACEV=JACEV+1
C
C     COMPUTE GN-SEARCH DIRECTION
C
      CALL GNSRCH(TIME,A,MDA,T,N,D1,P1,RANKA,NOHOUS,B,FMAT,MDF,C,MDC,
     1 M,F,PIVOT,TAU,MDG,SCALE,DIAG,INACT,Q,P4,P2,P3,DX,V1,D2,D3,
     2 RANKC2,D1NRM,DNRM,B1NRM,D,S,U,GMAT)
      I2=INACT(Q)
      I3=M+L
  200 CONTINUE
      RETURN
      END
