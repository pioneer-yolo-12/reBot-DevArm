#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
static int fake_fd=-1;
unsigned int if_nametoindex(const char *name) {
 if (!strcmp(name,"audit_mock_only")) return 123;
 return ((unsigned int(*)(const char*))dlsym(RTLD_NEXT,"if_nametoindex"))(name);
}
int socket(int domain,int type,int proto) {
 if(domain==29) { fake_fd=eventfd(0,EFD_NONBLOCK|EFD_CLOEXEC); return fake_fd; }
 return ((int(*)(int,int,int))dlsym(RTLD_NEXT,"socket"))(domain,type,proto);
}
int bind(int fd,const struct sockaddr *a,socklen_t len) {
 if(fd==fake_fd) return 0;
 return ((int(*)(int,const struct sockaddr*,socklen_t))dlsym(RTLD_NEXT,"bind"))(fd,a,len);
}
ssize_t write(int fd,const void *buf,size_t n) {
 if(fd==fake_fd) {
  const unsigned char *b=buf; char out[256]; int k=snprintf(out,sizeof out,"MOCK_CAN ");
  for(size_t i=0;i<n && k<220;i++)k+=snprintf(out+k,sizeof(out)-k,"%02x",b[i]);
  out[k++]='\n'; syscall(SYS_write,2,out,k); return n;
 }
 return syscall(SYS_write,fd,buf,n);
}
int close(int fd) { if(fd==fake_fd)fake_fd=-1;return syscall(SYS_close,fd); }
