(function () {
    'use strict';

    angular
        .module('app')
        .controller('HomeController', HomeController);

    HomeController.$inject = ['UserService', '$rootScope', '$http'];
    function HomeController(UserService, $rootScope, $http) {
        var vm = this;

        vm.user = null;
        vm.allUsers = [];
        vm.deleteUser = deleteUser;
        vm.rnResult = false;

        initController();

        function initController() {
            loadCurrentUser();
            loadAllUsers();
        }

        function loadCurrentUser() {
            UserService.GetByUsername($rootScope.globals.currentUser.username)
                .then(function (user) {
                    vm.user = user;
                });
        }

        function loadAllUsers() {
            UserService.GetAll()
                .then(function (users) {
                    vm.allUsers = users;
                });
        }

        function deleteUser(id) {
            UserService.Delete(id)
            .then(function () {
                loadAllUsers();
            });
        }

        vm.uploadFile = function uploadFile() {
            const f = document.getElementById('file').files[0];
            const fd = new FormData();
            fd.append('image', f, f.name);
            $http.post('/upload', fd, {
                transformRequest: angular.identity,
                headers: {'Content-Type': undefined}
            }).then(function success(response) {
                console.log(response);
                vm.rnResult = response.data;
            }, function error(err) {
                console.error(err);
            });
        }
    }

})();