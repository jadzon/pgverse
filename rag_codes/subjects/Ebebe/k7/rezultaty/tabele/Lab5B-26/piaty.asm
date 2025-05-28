; zad5 - FIR filter with input from file (e.g., < data.txt)
; Modified: load 64 numbers into WE array, then apply FIR:
;   wy[0] = x0, wy[1] = x1, wy[2] = x2
;   wy[i] = (120*x[i-1])/128 - (200*x[i-2])/512 + (60*x[i-3]+210)/256
; Parameters passed to FIR via stack. Display in decimal.

dane SEGMENT
    cyfrywe dw 5
    cyfrywy dw 5
    we      dw 64 dup(?)  ; input array
    wy      dw 64 dup(?)  ; output array
    NL      db 13,10,'$'
dane ENDS

rozkazy SEGMENT 'CODE' use16
ASSUME cs:rozkazy, ds:dane

startuj:
    mov ax, SEG dane
    mov ds, ax

    mov cx, 64
    xor si, si           ; SI = 0

; --- Read 64 numbers into WE ---
wczytaj:
    lea bx, [we + si]
    push bx             ; parameter: pointer to we[i]
    mov ax, cyfrywe
    push ax             ; max digits
    call WczyLiczbe10
    add sp, 4

    add si, 2
    loop wczytaj

; --- Call FIR: FIR(len, wy_ptr, we_ptr) ---
    push 64
    lea bx, wy
    push bx
    lea bx, we
    push bx
    call FIR
    add sp, 6

; --- Display WE array ---
    mov cx, 64
    xor si, si
wyswietl_we:
    lea bx, [we + si]
    push bx             ; pointer to value
    mov ax, cyfrywy
    push ax             ; max digits
    call WyswLiczbe10
    add sp, 4
    call NowaLinia

    add si, 2
    loop wyswietl_we

; --- Display WY array ---
    mov cx, 64
    xor si, si
wyswietl_wy:
    lea bx, [wy + si]
    push bx
    mov ax, cyfrywy
    push ax
    call WyswLiczbe10
    add sp, 4
    call NowaLinia

    add si, 2
    loop wyswietl_wy

    call koniec

;-----------------------------------------------
FIR PROC
    push bp
    mov bp, sp
    push si
    push di
    push bx

    mov cx, [bp+6]      ; length
    mov si, [bp+8]      ; we pointer
    mov di, [bp+4]      ; wy pointer

    ; Copy first three samples
    mov ax, [si]
    mov [di], ax
    add si, 2
    add di, 2

    mov ax, [si]
    mov [di], ax
    add si, 2
    add di, 2

    mov ax, [si]
    mov [di], ax
    add si, 2
    add di, 2

    mov bx, 3
petla:
    ; dx = (120*we[i-1])/128
    mov ax, [si-2]
    mov bx, 120
    imul bx
    sar ax, 7
    mov dx, ax

    ; dx -= (200*we[i-2])/512
    mov ax, [si-4]
    mov bx, 200
    imul bx
    sar ax, 9
    neg ax
    add dx, ax

    ; dx += (60*we[i-3] + 210)/256
    mov ax, [si-6]
    mov bx, 60
    imul bx
    add ax, 210
    shr ax, 8
    add dx, ax

    mov [di], dx
    add si, 2
    add di, 2
    inc bx
    cmp bx, cx
    jl petla

    pop bx
    pop di
    pop si
    pop bp
    ret 6
FIR ENDP

;-----------------------------------------------
WczyLiczbe10 PROC
    push bp
    mov bp, sp
    push cx
    push bx
    push dx
    push si

    mov si, [bp+6]      ; pointer
    xor ax, ax
    mov [si], ax        ; clear

    mov cx, [bp+4]      ; digits count

    ; Skip leading whitespace
skip_whitespace:
    mov ah, 01h         ; Read character with echo from standard input
    int 21h
    cmp al, ' '         ; space
    je skip_whitespace
    cmp al, 13          ; carriage return
    je skip_whitespace
    cmp al, 10          ; line feed
    je skip_whitespace
    cmp al, 9           ; tab
    je skip_whitespace
    
    ; Check if it's a digit
    cmp al, '0'
    jb end_number
    cmp al, '9'
    ja end_number
    
    ; Process first digit
    sub al, '0'
    mov bl, al
    xor bh, bh
    mov [si], bx
    dec cx
    jz end_number

czn:
    mov ah, 01h         ; Read character with echo from standard input
    int 21h
    
    ; Check for end of number
    cmp al, ' '         ; space
    je end_number
    cmp al, 13          ; carriage return
    je end_number
    cmp al, 10          ; line feed
    je end_number
    cmp al, 9           ; tab
    je end_number
    cmp al, 26          ; EOF (Ctrl+Z)
    je end_number
    
    ; Check if it's a digit
    cmp al, '0'
    jb end_number
    cmp al, '9'
    ja end_number
    
    sub al, '0'
    mov bl, al
    xor bh, bh
    mov ax, 10
    mul word ptr [si]
    add ax, bx
    mov [si], ax
    loop czn

end_number:
    pop si
    pop dx
    pop bx
    pop cx
    pop bp
    ret
WczyLiczbe10 ENDP

;-----------------------------------------------
WyswLiczbe10 PROC
    push bp
    mov bp, sp
    push ax
    push bx
    push cx
    push dx

    mov si, [bp+6]      ; pointer
    mov ax, [si]
    xor cx, cx
    cmp ax, 0
    jne WLDZ_div
    mov dl, '0'
    mov ah, 02h
    int 21h
    jmp WLDZ_done
WLDZ_div:
    xor dx, dx
    mov bx, 10
WLDZ_loop:
    div bx
    push dx
    inc cx
    xor dx, dx
    cmp ax, 0
    jne WLDZ_loop
WLDZ_prn:
    pop dx
    add dl, '0'
    mov ah, 02h
    int 21h
    loop WLDZ_prn
WLDZ_done:
    pop dx
    pop cx
    pop bx
    pop ax
    pop bp
    ret
WyswLiczbe10 ENDP

;-----------------------------------------------
NowaLinia PROC
    push dx
    mov dx, OFFSET NL
    mov ah, 09h
    int 21h
    pop dx
    ret
NowaLinia ENDP

koniec PROC
    mov ax, 4C00h
    int 21h
koniec ENDP

rozkazy ENDS

stosik SEGMENT STACK
    dw 128 dup(?)
stosik ENDS

END startuj
