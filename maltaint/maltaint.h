/*
 * taintgrind.h
 *
 *  Created on: Jun 12, 2012
 *      Author: khilan
 */

#ifndef TAINTGRIND_H_
#define TAINTGRIND_H_

#include "valgrind.h"
#include <sys/syscall.h>
#include <stdio.h>


typedef enum {
	VG_USERREQ__NONE_CREATE_SANDBOX,
    VG_USERREQ__NONE_CREATE_NOEFFSET,
    VG_USERREQ__NONE_CREATE_EFFSET,
	VG_USERREQ__NONE_TAINT,
    VG_USERREQ__MALTAINT_CALL_OPEN
} Vg_TaintGrindClientRequest;


// Tainting/Untainting memory
#define TNT_PRINTSEC() \
        VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__NONE_CREATE_NOEFFSET,0,0,0,0,0);\

#define TNT_PRINTFIRST() \
        VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__NONE_CREATE_EFFSET,0,0,0,0,0); \

#define TNT_TAINT(addr, size) \
		VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__NONE_TAINT,addr,size,0,0,0); \

#define TNT_PRINT() \
		VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__NONE_CREATE_SANDBOX,0,0,0,0,0); \


#undef DO_CREQ_v_W
#undef DO_CREQ_W_W
#undef DO_CREQ_v_WW
#undef DO_CREQ_W_WW
#undef DO_CREQ_v_WWW
#undef DO_CREQ_W_WWW
#undef DO_CREQ_v_WWWW
#undef DO_CREQ_v_WWWWW

#define DO_CREQ_v_W(_creqF, _ty1F,_arg1F)                \
    do {                                                  \
        long int _arg1;                                    \
        _arg1 = (long int)(_arg1F);                        \
        VALGRIND_DO_CLIENT_REQUEST_STMT(                   \
                (_creqF),               \
                _arg1, 0,0,0,0);        \
    } while (0)

#define DO_CREQ_W_W(_resF, _dfltF, _creqF, _ty1F,_arg1F) \
    do {                                                  \
        long int _arg1;                                    \
        _arg1 = (long int)(_arg1F);                        \
        _qzz_res = VALGRIND_DO_CLIENT_REQUEST_EXPR(        \
                (_dfltF),               \
                (_creqF),               \
                _arg1, 0,0,0,0);        \
        _resF = _qzz_res;                                  \
    } while (0)

#define DO_CREQ_v_WW(_creqF, _ty1F,_arg1F, _ty2F,_arg2F) \
    do {                                                  \
        long int _arg1, _arg2;                             \
        _arg1 = (long int)(_arg1F);                        \
        _arg2 = (long int)(_arg2F);                        \
        VALGRIND_DO_CLIENT_REQUEST_STMT(                   \
                (_creqF),               \
                _arg1,_arg2,0,0,0);     \
    } while (0)

#define DO_CREQ_v_WWW(_creqF, _ty1F,_arg1F,              \
        _ty2F,_arg2F, _ty3F, _arg3F)       \
do {                                                  \
    long int _arg1, _arg2, _arg3;                      \
    _arg1 = (long int)(_arg1F);                        \
    _arg2 = (long int)(_arg2F);                        \
    _arg3 = (long int)(_arg3F);                        \
    VALGRIND_DO_CLIENT_REQUEST_STMT(                   \
            (_creqF),               \
            _arg1,_arg2,_arg3,0,0); \
} while (0)

#define DO_CREQ_W_WWW(_resF, _dfltF, _creqF, _ty1F,_arg1F, \
        _ty2F,_arg2F, _ty3F, _arg3F)       \
do {                                                  \
    long int _qzz_res;                                 \
    long int _arg1, _arg2, _arg3;                      \
    _arg1 = (long int)(_arg1F);                        \
    _arg2 = (long int)(_arg2F);                        \
    _arg3 = (long int)(_arg3F);                        \
    _qzz_res = VALGRIND_DO_CLIENT_REQUEST_EXPR(        \
            (_dfltF),               \
            (_creqF),               \
            _arg1,_arg2,_arg3,0,0); \
    _resF = _qzz_res;                                  \
} while (0)

#define DO_CREQ_v_WWWW(_creqF, _ty1F,_arg1F,             \
        _ty2F, _arg2F, _ty3F, _arg3F,     \
        _ty4F, _arg4F)                    \
do {                                                  \
    Word _arg1, _arg2, _arg3, _arg4;                   \
    _arg1 = (Word)(_arg1F);                            \
    _arg2 = (Word)(_arg2F);                            \
    _arg3 = (Word)(_arg3F);                            \
    _arg4 = (Word)(_arg4F);                            \
    VALGRIND_DO_CLIENT_REQUEST_STMT((_creqF),          \
            _arg1,_arg2,_arg3,_arg4,0); \
} while (0)

#define DO_CREQ_v_WWWWW(_creqF, _ty1F,_arg1F,        \
        _ty2F, _arg2F, _ty3F, _arg3F,     \
        _ty4F, _arg4F, _ty5F, _arg5F)     \
do {                                                 \
    long int _arg1, _arg2, _arg3, _arg4, _arg5;        \
    _arg1 = (long int)(_arg1F);                        \
    _arg2 = (long int)(_arg2F);                        \
    _arg3 = (long int)(_arg3F);                        \
    _arg4 = (long int)(_arg4F);                        \
    _arg5 = (long int)(_arg5F);                        \
    VALGRIND_DO_CLIENT_REQUEST_STMT(                   \
            (_creqF),                       \
            _arg1,_arg2,_arg3,_arg4,_arg5); \
} while (0)



#endif /* TAINTGRIND_H_ */
