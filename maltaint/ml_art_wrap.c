
// // #include "pub_tool_basics.h"
// // #include "valgrind.h"
// // #include "ml_include.h"

// // #define  BG_Z_LIBART_SONAME  libartZdsoZa              // libart.so*
// // #define LIBART_FUNC(ret_ty, f, args...) \
// //     ret_ty I_WRAP_SONAME_FNNAME_ZU(BG_Z_LIBART_SONAME,f)(args); \
// //     ret_ty I_WRAP_SONAME_FNNAME_ZU(BG_Z_LIBART_SONAME,f)(args)

// // void* DexFile_DexFile(void *this, void *base, int size, void* location, int checksum, void* mem_map, void* oat_dex_file)
// // {
// //     OrigFn fn;
// //     void* res = NULL;
// // TNT_PRINT();
// //     //DO_CREQ_v_WWWW(VG_USERREQ__WRAPPER_ART_DEXFILE_PRE, void*, mem_map, char*, base, int, size, char*, (char*)(*((unsigned int*)location+2)));
// //     VALGRIND_GET_ORIG_FN(fn);
// //     CALL_FN_W_7W(res, fn, this, base, size, location, checksum, mem_map, oat_dex_file);
// //      DO_CREQ_v_WWWWW(VG_USERREQ__NONE_CREATE_SANDBOX, Addr, (Addr)this, char*, base, int, size, void*, location, void*, mem_map);
// //     return res;
// // }
// // // _ZN3art7DexFileC2EPKhjRKNSt3__112basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEEjPNS_6MemMapEPKNS_10OatDexFileE
// // LIBART_FUNC(void*, _ZN3art7DexFileC2EPKhjRKNSt3__112basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEEjPNS_6MemMapEPKNS_10OatDexFileE,
// //         void* this, void* base, int size, void* location, int checksum, void* mem_map, void* oat_dex_file)
// // {
// // 	TNT_PRINT();
// //     return DexFile_DexFile(this, base, size, location, checksum, mem_map, oat_dex_file);
// // }


#include "pub_tool_basics.h"
#include "pub_tool_redir.h"
#include "maltaint.h"

#include "valgrind.h"

#include <stdio.h> // printf when testing
#include <inttypes.h>
#define VG_Z_LIBART_SONAME libartZdsoZa
#define VG_Z_LIBC_SONAME libcZdZa
#define W_FUNC(L,ret_ty, f, args...) \
   ret_ty I_WRAP_SONAME_FNNAME_ZU(L,f)(args); \
   ret_ty I_WRAP_SONAME_FNNAME_ZU(L,f)(args)

#define W_LIBC_FUNC(ret_ty, f, args...) W_FUNC(VG_Z_LIBC_SONAME,ret_ty,f,args)
#define W_LIBART_FUNC(ret_ty, f, args...) W_FUNC(VG_Z_LIBART_SONAME,ret_ty,f,args)

// //OatFile::OatDexFile::GetOatClassOffset
// W_LIBART_FUNC(int,_ZNK3art10OatDexFile17GetOatClassOffsetEt,void* this, uint16_t class_def_index) {
//     TNT_PRINTFIRST();
//     int fd = -1;
//     OrigFn fn;
//     VALGRIND_GET_ORIG_FN(fn);
//     CALL_FN_W_W(fd, fn,this,class_def_index);
//     return fd;
// }  

//OatFile::OatDexFile::GetOatClass
W_LIBART_FUNC(int,_ZNK3art10OatDexFile11GetOatClassEt,void* this, uint16_t class_def_index) {
    
    int fd = -1;
    OrigFn fn;
    VALGRIND_GET_ORIG_FN(fn);
    CALL_FN_W_WW(fd, fn,this,class_def_index);
    TNT_PRINTSEC();
    return fd;
}  
// // _ZN3art7OatFile4OpenERKNSt3__112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES9_PhSA_bbPKcPS7_
 W_LIBART_FUNC(int,
     _ZN3art7OatFile4OpenERKNSt3__112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES9_PhSA_bbPKcPS7_,
     void* this,
     char* oat_filename,
     char* oat_location,
     uint8_t* requested_base,
     uint8_t* oat_file_begin,
     Bool executable,
     Bool low_4gb,
     const char* abs_dex_location,char* error_msg){
     int fd = -1 ;
     OrigFn fn;
     VALGRIND_GET_ORIG_FN(fn);
     TNT_PRINTSEC();
     CALL_FN_W_9W(fd, fn,this,oat_filename,oat_location,requested_base,oat_file_begin,executable,low_4gb,abs_dex_location,error_msg);
     DO_CREQ_v_WWW(VG_USERREQ__MALTAINT_CALL_OPEN,void*,this,char*,oat_filename,char*,oat_location);
     return fd;
 }  

//_ZN3art7OatFile12FindOatClassERKNS_7DexFileEtPb
   // OatFile::FindOatClass(const DexFile& dex_file,
   //                                      uint16_t class_def_idx,
   //                                      bool* found)

// W_LIBART_FUNC(void*,_ZN3art7OatFile12FindOatClassERKNS_7DexFileEtPb, void* dex_file，int class_def_idx, char found) {
    
//     int    fd = -1;
//     OrigFn fn;
//     VALGRIND_GET_ORIG_FN(fn);
//     TNT_PRINTSEC();
//     // CALL_FN_W_W(fd, fn,class_def_index);
//     return fd;
// } 

W_LIBC_FUNC(int, open, const char *path, int oflags) {
    
    int fd = -1;
    OrigFn fn;
    VALGRIND_GET_ORIG_FN(fn);
    // CALL_FN_W_WW(fd, fn, path, oflags);
    TNT_PRINT();
    return fd;
}
