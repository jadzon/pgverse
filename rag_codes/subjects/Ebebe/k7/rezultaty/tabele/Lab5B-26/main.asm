;——————————————————————————————————————————————————————————————
; Program: FIR_from_file.asm
; Assembler: MASM/TASM (16-bit real mode, DOS)
; Opis:      – wczytuje dane z "dane.txt" (maks. 64 wartości heksadecymalne)
;            – filtr FIR wg podanego wzoru
;            – wyświetla tablice we i wy w systemie dziesiętnym
;——————————————————————————————————————————————————————————————
.386
.model small
.stack 100h

.data
    filename    db 'dane.txt',0
    fileHandle  dw ?
    buf         db 128           ; bufor dla ReadFile
    bytesRead   dw ?
    we          dw 64 dup(?)     ; tablica wejściowa
    wy          dw 64 dup(?)     ; tablica wyjściowa
    count       dw 0             ; ile wczytano wartości
    msgErrOpen  db 'Blad otwarcia pliku!',0Dh,0Ah,'$'
    msgErrRead  db 'Blad odczytu pliku!',0Dh,0Ah,'$'
    msgWe       db 'Tablica we:',0Dh,0Ah,'$'
    msgWy       db 'Tablica wy:',0Dh,0Ah,'$'
    newline     db 0Dh,0Ah,'$'

.code
start:
    mov ax,@data
    mov ds,ax

    ;———————————
    ; 1) Otwarcie pliku dane.txt
    lea  dx,filename
    mov  ah,3Dh        ; Open file
    xor  al,al         ; tylko do odczytu
    int 21h
    jc   ErrOpen
    mov  [fileHandle],ax  ; Store file handle returned in AX

    ;———————————
    ; 2) Wczytanie i parsowanie do we[]
    lea  si,buf        ; Use LEA instead of OFFSET
    mov  bx,0          ; BX = indeks w we[]
    mov  di,0          ; DI = stan parsowania (0=czekaj na hex,1=nibble hi)
    mov  cx,0          ; CX = akumulator dla pary hex
ReadLoop:
    ; czytamy blok
    mov  ah,3Fh        ; Read File
    mov  bx,[fileHandle] ; This was changed above, ensure it's word ptr if error persists here too
    mov  cx,128
    lea  dx,buf
    int 21h
    jc   ErrRead
    mov  [bytesRead], ax
    cmp  ax,0
    je   AfterParse   ; koniec pliku

    lea  si, buf
    mov  ax,[bytesRead]
    mov  dx,ax         ; DX = liczb bajtów w buforze


ByteLoop:
    mov  al,[si]
    ; sprawdź czy cyfra hex
    call  isHexChar    ; AL = wartość 0..15 lub 0FFh jeśli nie-hex
    cmp   al,0FFh
    je    SkipChar
    ; mamy nibble
    cmp   di,0
    je    SaveHighNibble
    ; tu: di=1, więc to nibble low
    mov   bl,al        ; BL = low nibble
    shl   cx,4         ; stary hi w CX[3:0] -> CX[7:4]
    and   cx,0FFh
    or    cx,bx        ; CX = pełny bajt

    ; zapisz do we[]
    cmp   bx,64
    jae   SkipStore    ; tylko pierwsze 64 wartości
    mov   ax,cx
    ; mov   [we+bx*2],ax ; Original problematic line
    mov   di, bx       ; Use DI for byte offset calculation
    shl   di, 1        ; DI = bx * 2 (byte offset for word array)
    mov   [we+di], ax  ; Store word AX at we[bx]
    inc   bx

SkipStore:
    xor   cx,cx        ; wyczyść akumulator
    mov   di,0         ; reset stanu
    jmp   NextChar

SaveHighNibble:
    mov   cx,al        ; CX = hi nibble
    mov   di,1         ; czekamy na low
    jmp   NextChar

SkipChar:
    ; jeśli separator: jeżeli w połowie pary, zerujemy stan
    mov   di,0
    xor   cx,cx

NextChar:
    inc   si
    dec   dx
    jnz   ByteLoop
    jmp   ReadLoop

AfterParse:
    mov   word ptr [count],bx

    ; zamknięcie pliku
    mov   ah,3Eh
    mov   bx,[fileHandle]
    int   21h

    ; jeżeli <3 wartości, idziemy do końca
    mov   ax,[count]
    cmp   ax,3
    jl    DisplayData

    ;———————————
    ; 3) Wywołanie filtru FIR
    push word ptr [count]
    mov  ax, offset wy         ; Use OFFSET instead of LEA
    push ax                    ; Push 16-bit offset
    mov  ax, offset we         ; Use OFFSET instead of LEA  
    push ax                    ; Push 16-bit offset
    call FIR
    add  sp,6

DisplayData:
    ;———————————
    ; 4) Wyświetlenie tablicy we
    lea  dx,msgWe
    mov  ah,09h
    int 21h
    mov  cx,[count]
    lea  si,we
PrintWe:
    mov  ax,[si]
    call WyswLiczbe10
    lea  dx,newline
    mov  ah,09h
    int 21h
    add  si,2
    loop PrintWe

    ;———————————
    ; 5) Wyświetlenie tablicy wy
    lea  dx,msgWy
    mov  ah,09h
    int 21h
    mov  cx,[count]
    lea  si,wy
PrintWy:
    mov  ax,[si]
    call WyswLiczbe10
    lea  dx,newline
    mov  ah,09h
    int 21h
    add  si,2
    loop PrintWy

    ;———————————
    ; 6) Zakończenie
    mov ah,4Ch
    int 21h

;——————————————————————————————————————————————————————————————
; FIR – filtr FIR wg wzoru
; Parametry: [bp+4]=offset we, [bp+6]=offset wy, [bp+8]=długość (word)
;——————————————————————————————————————————————————————————————
FIR proc near
    push bp
    mov  bp,sp
    mov  ax,[bp+8]
    mov  cx,ax
    mov  si,[bp+4]
    mov  di,[bp+6]

    ; kopiujemy pierwsze 3 próbki
    mov  dx,3
CPY0:
    mov  ax,[si]
    mov  [di],ax
    add  si,2
    add  di,2
    dec  dx
    jnz CPY0

    ; dla i=3..len-1 obliczamy
    mov  ax,[bp+8]
    sub  ax,3
    mov  cx,ax
LOOPF:
    ; term1 = (120*we[i-1])>>7
    mov  bx,si
    sub  bx,2
    mov  ax,[bx]
    imul ax,120
    sar  ax,7
    mov  bp,ax

    ; term2 = -((200*we[i-2])>>9)
    mov  bx,si
    sub  bx,4
    mov  ax,[bx]
    imul ax,200
    sar  ax,9
    neg  ax
    add  bp,ax

    ; term3 = ((60*we[i-3]+210)>>8)
    mov  bx,si
    sub  bx,6
    mov  ax,[bx]
    imul ax,60
    add  ax,210
    shr  ax,8
    add  bp,ax

    mov  [di],bp

    add  si,2
    add  di,2
    loop LOOPF

    pop bp
    ret 6
FIR endp

;——————————————————————————————————————————————————————————————
; isHexChar: AL=znak ASCII, zwraca w AL wartość 0..15 lub 0FFh jeśli nie-hex
;——————————————————————————————————————————————————————————————
isHexChar proc near
    push ax
    cmp  al,'0'
    jl   NotHex
    cmp  al,'9'
    jle  ToValDigit
    cmp  al,'A'
    jl   NotHex
    cmp  al,'F'
    jle  ToValUpper
    cmp  al,'a'
    jl   NotHex
    cmp  al,'f'
    jle  ToValLower
NotHex:
    mov  al,0FFh
    pop  bx
    ret
ToValDigit:
    sub  al,'0'
    pop  bx
    ret
ToValUpper:
    sub  al,'A'-10
    pop  bx
    ret
ToValLower:
    sub  al,'a'-10
    pop  bx
    ret
isHexChar endp

;——————————————————————————————————————————————————————————————
; WyswLiczbe10 – wyświetla zawartość AX w systemie dziesiętnym
;——————————————————————————————————————————————————————————————
WyswLiczbe10 proc near
    push ax bx cx dx
    cmp  ax,0
    jne  WLZ_NZ
    mov  dl,'0'
    mov  ah,02h
    int 21h
    jmp  WLZ_DONE
WLZ_NZ:
    xor  cx,cx
WLZ_DIV:
    xor  dx,dx
    mov  bx,10
    div  bx
    push dx
    inc  cx
    cmp  ax,0
    jne  WLZ_DIV
WLZ_PRT:
    pop  dx
    add  dl,'0'
    mov  ah,02h
    int 21h
    dec  cx
    jnz  WLZ_PRT
WLZ_DONE:
    pop  dx cx bx ax
    ret
WyswLiczbe10 endp

ErrOpen:
    lea dx,msgErrOpen
    mov ah,09h
    int 21h
    jmp ExitProg

ErrRead:
    lea dx,msgErrRead
    mov ah,09h
    int 21h
    jmp ExitProg

ExitProg:
    mov ah,4Ch
    int 21h

end start
